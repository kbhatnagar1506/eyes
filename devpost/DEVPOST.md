# 👁 Eyes: cameras that watch. Eyes that act.

> **Every city is already covered in cameras. Not one of them has ever called for help.**
> Eyes turns the cameras we already have into a safety network that **detects** danger, **dispatches** help and **warns the people nearby**, even when the internet is down, all coordinated by AI agents that live in iMessage.

![Eyes](https://raw.githubusercontent.com/kbhatnagar1506/eyes/main/devpost/gallery/01-eyes.jpg)

## 🔗 Try it

| | |
|---|---|
| **Live app** | https://eyes-hacklanta.vercel.app (Dataset tool · Live demo · Mesh · Model) |
| **The model, the data, the infrastructure** | https://eyes-hacklanta.vercel.app → **Model** tab |
| **iMessage agents** | Built on Photon Spectrum; come to our table and text the Eyes line |
| **Code** | https://github.com/kbhatnagar1506/eyes |

---

## 💡 Inspiration

Imagine your sister walking home after dark.

A man starts following her. She speeds up; so does he. She crosses the street; he crosses too. She runs, and he runs after her. She passes a bus stop, a store, a parking deck, and at every one of them a camera turns its glass eye toward her and **watches**. It records every second of the worst night of her life.

And it does nothing.

It doesn't call anyone. It doesn't warn the people a block away. It doesn't tell the bus driver pulling up to the stop. Tomorrow someone might pull the footage to figure out what happened, but tonight the camera is just a witness that never speaks.

That's the part that stuck with us. **The infrastructure to protect her already exists. It's bolted to every wall she ran past.** It just has no brain and no voice.

So we gave it both.

---

## ⚡ What it does

Eyes is a real-time safety network for public places, built in four layers:

| | Layer | What happens |
|---|---|---|
| 👁 | **Detect** | A camera sees a situation turn violent. AttackNet, our model, reads the body language of everyone in the frame, and a box turns red around the people involved. The evidence frame is captured that instant. |
| 🕸 | **Spread** | The incident spreads across a **mesh of 130 city nodes**: police, hospitals, MARTA stations and live buses. A networking agent sends every message over whatever link works (internet, P2P Wi-Fi, or mesh radio), so the alert gets out **even offline**. |
| 💬 | **Coordinate** | A team of AI agents takes over in **iMessage**: they page the nearest responders, send police the camera frame, track who is actually on the way, and keep one calm live status card updated for everyone. |
| 🛡 | **Protect** | Anyone within 800 m who signed up gets a loud alert ("⚠️ be careful around Woodruff Park"), a safe spot to go to and an "Are you safe?" poll. Anyone hurt gets first aid by text until EMS arrives. And **anyone can report** an incident just by texting what's happening and where. |

The camera stops being a witness and becomes a first responder.

---

## 🧭 How it works, end to end

```
                    ┌──────────────── 1. DETECT ─────────────────┐
  CCTV / webcam ──▶ │ YOLOv8x-pose → 84 body-language signals →  │
                    │ AttackNet (1.9M weights) → p(attack) per   │
                    │ person · box turns red above 30%           │
                    └───────────────┬────────────────────────────┘
                                    │ incident + evidence frame
                    ┌───────────────▼──────── 2. SPREAD ─────────┐
                    │ Eyes mesh: HLC clock + last-writer-wins    │
                    │ store, gossip, 48 KB chunks                │
                    │ Networking agent picks the best link:      │
                    │ same device · P2P Wi-Fi · mesh radio ·     │
                    │ internet relay  →  130 city nodes          │
                    └───────┬───────────────────────┬────────────┘
                            │                       │
           ┌────────────────▼───────┐   ┌───────────▼──────────────────┐
           │ 3a. RESPONDER PHONES   │   │ 3b. DISPATCH AI (agents)     │
           │ officer · bus driver · │   │ Sentinel · Network ·         │
           │ rider: frame, map,     │   │ Dispatch · Police · EMS ·    │
           │ Respond / Injured /    │   │ Safety · Care · Intake       │
           │ Resolved               │   │ Gemini 2.5 Flash on Vertex   │
           └────────────────────────┘   └───────────┬──────────────────┘
                                                    │ Photon Spectrum
                    ┌───────────────────────────────▼────────────┐
                    │ 4. iMESSAGE                                │
                    │ responders: live status card (edited in    │
                    │   place) · threaded brief · police photos  │
                    │ public: area alerts · polls · reports ·    │
                    │   care · all clear                         │
                    └───────────────┬────────────────────────────┘
                                    ▼
                  Cloud SQL Postgres + Cloud Storage (the record)
```

![Response flow](https://raw.githubusercontent.com/kbhatnagar1506/eyes/main/devpost/gallery/13-flow.jpg)

---

## 🧠 Technical depth: the model, the data, and how it works

### The model: AttackNet

Most "fight detectors" stare at pixels. Pixels are a terrible way to understand a fight: they carry lighting, clothing, camera angle and faces, and none of that is what violence *is*. Violence is **how bodies move relative to each other**: one person closing in, a limb moving fast, another person's arms coming up to shield their head, a crowd converging or scattering.

So AttackNet never sees pixels. It sees **skeletons**.

```
 video (10 fps)
   │
   ▼
 YOLOv8x-pose ── every person: box + 17 keypoints ── IoU tracker keeps identities
   │
   ▼
 84 body-language features per person per frame
   │   posture: lean, someone down, crouch
   │   strikes: wrist / ankle speed relative to the body
   │   defense: arms raised, hands shielding the head
   │   contact: distance to nearest person, closing in, limb inside their box
   │   crowd:   people converging or fleeing, crowd size
   ▼
 window: 8 people × 20 frames (2 s) × 84
   │
   ▼
 Input projection        LayerNorm + Linear 84 → 160              13,768 weights
 Temporal transformer ×4 each person's last 2 s, CLS, 4 heads  1,237,120 weights
 People transformer ×2   everyone in the frame attends to       618,560 weights
                         everyone else
 Head                    MLP → p(attack) for each person           26,241 weights
   │
   ▼
 Multiple-instance learning: frame = its most violent person,
                             clip  = its most violent moments
```

**1,899,209 weights.** Small on purpose: for ~2,000 training clips, a bigger model memorizes videos instead of learning patterns.

Why two transformers? A punch is a *sequence* (tension builds, someone lunges, someone recoils), so the **temporal transformer** reads each person's last two seconds. But a fight only makes sense *between* people: an attacker, a victim and a bystander can look identical alone. So the **people transformer** lets every person in the frame attend to everyone else.

**The labeling trick.** Real datasets say *when* a fight happens, never *who* is fighting. So we train with **multiple-instance learning**: a frame's score is its most violent person, and a clip's score is its most violent moments,

$$
p_{\text{frame}} = \max_{i \in \text{people}} p_i, \qquad p_{\text{clip}} = \frac{1}{|K|}\sum_{t \in K} \tilde{p}_t, \quad K = \text{top 25\% of smoothed moments}
$$

and the per-person scores learn *who* is involved on their own. That's why, on the CCTV clip above, the red box lands on the right person even though no one ever told the model which person it was.

### The data

| Set | What | How we use it |
|---|---|---|
| **Real Life Violence Situations** | 1,000 fight + 1,000 normal real-world clips | 70% train · 10% thresholds · **20% test, never trained on** (split by clip) |
| **Our CCTV + demo clips** | 8 clips (UCF-Crime fights, assaults and normal footage, a calm bus stop, our own camera), **labeled frame by frame** by hand | Scored one at a time by a model that never saw them |
| **Sparring clips** | 4 clips | Kept out of training entirely: the hardest test, because sparring *looks* like fighting |

All of it ran on a **Google Cloud NVIDIA T4**: YOLOv8x-pose in half precision followed **29,869 person tracks across 98,000 frames**, then AttackNet trained in about 10 minutes. The VM is created per run and deleted after.

### The results, honestly

![Results](https://raw.githubusercontent.com/kbhatnagar1506/eyes/main/devpost/gallery/09-results.jpg)

| Test (never trained on) | v1 (our clips only) | **v2 (pretrained on RLVS)** |
|---|---|---|
| 400 held-out real-world clips | not tested | **91.5% accuracy · AUC 0.975** |
| Our hand-labeled CCTV | F1 56% · AUC 0.64 | **F1 63% · AUC 0.72** |
| Showcase clip (CCTV assault) | | **F1 83%** |
| Sparring | | F1 61% · AUC 0.51 |

On the 400 test clips it got 185 of 200 fights and 181 of 200 normal clips right. Every number on our Model page is on footage the model **never trained on**. We also tried fine-tuning on our CCTV; it scored *worse* on unseen clips, so **we didn't ship it**, and the page says so. Sparring is still hard (it genuinely looks like fighting), and we show that too.

![Training](https://raw.githubusercontent.com/kbhatnagar1506/eyes/main/devpost/gallery/10-training.jpg)

### The live detector: StrikeDetector

For the live demo, a second detector runs **entirely in the browser**, on the laptop itself: MediaPipe pose (33 landmarks, up to 4 people) feeding a geometric hit model. Each person gets a torso capsule and a head circle as a hitbox, and a strike fires when one person's hand enters another person's hitbox fast enough:

$$
v_{\text{hand}} = \frac{\lVert \Delta(\mathbf{w} - \mathbf{c}) \rVert}{L_{\text{torso}}\,\Delta t} > 3 \ \text{torso-lengths/s}
$$

Speed is measured **relative to the body** (\\(\mathbf{w}\\) wrist, \\(\mathbf{c}\\) torso center) and in **torso lengths** (\\(L_{\text{torso}}\\)), so it works for anyone at any distance from the camera. Glitch rejection, warm-up and per-pair cooldowns stop false alarms. A punch from a judge on camera is caught in the same frame, with no server round trip.

### The mesh

The alert has to get out even when the internet doesn't. We built our own replicated store instead of using an SDK:

- **Hybrid logical clocks + field-level last-writer-wins.** Every device can write; every device converges to the same incident.
- **Gossip of fresh ops only**, in 48 KB chunks, so a large evidence frame never blocks the tiny alert ahead of it.
- **A networking agent** pings every link (round-trip time and loss) and sends each message over the best one, **with the reason**, then fails over and resyncs with a snapshot.
- **The Atlanta mesh:** 130 city nodes, 281 radio links, 43 live MARTA buses (GTFS-realtime), 2,155 public places, and 7,011 stops on 169 routes.

![Mesh](https://raw.githubusercontent.com/kbhatnagar1506/eyes/main/devpost/gallery/03-spread.jpg)

---

## 🏗 Architecture

![Infrastructure](https://raw.githubusercontent.com/kbhatnagar1506/eyes/main/devpost/gallery/12-infrastructure.jpg)

| Layer | Built with |
|---|---|
| Edge | MediaPipe pose + StrikeDetector in the browser · responder screens join by QR |
| Mesh | Our own HLC + LWW replicated store · WebRTC P2P · simulated mesh radio · Bun WebSocket hub behind a Cloudflare tunnel |
| Agents | Bun · Gemini 2.5 Flash on Vertex AI · Photon Spectrum (iMessage) |
| Model | YOLOv8x-pose · PyTorch AttackNet · DensePose (Detectron2) renders · NVIDIA T4 on Compute Engine |
| Data | Cloud SQL Postgres (incidents, acknowledgements, agent messages, evidence, sign-ups) · Cloud Storage (private, 7-day signed links) |
| Web | Vite + TypeScript · Leaflet + OpenStreetMap · Chart.js · Vercel (app + live-bus function) |

---

## 🏆 Sponsor tracks

### 🚗 Mercedes-Benz: Best Application of Computer Vision

Eyes is computer vision that **acts**. Three vision systems, each picked for its job:

1. **AttackNet (the main model).** YOLOv8x-pose → 84 body-language signals → temporal + people transformers, trained on a T4. **91.5% / AUC 0.975 on 400 unseen real-world clips; 83% F1 on the showcase CCTV clip it never saw.** Red boxes appear only above 30%, so a calm scene stays green.
2. **StrikeDetector (live).** MediaPipe in the browser with a geometric hit model: a punch is caught *in the same frame* it lands, on a laptop, with no GPU and no server.
3. **DensePose on the T4.** Every visible body's surface mapped and tracked across frames: the dense body model the Dataset tool plays back.

What makes it a real application and not a demo: it's **honest** (every score is on unseen footage, including the failures), **privacy-first** (it reads 17 skeleton points, never faces or identity), and it's wired to the real world. The frame that turns red is the same frame an officer receives on their phone two seconds later.

![Detect](https://raw.githubusercontent.com/kbhatnagar1506/eyes/main/devpost/gallery/02-detect.jpg)

### 💬 Photon: Agents in iMessage

When the camera sees something, **a team of AI agents takes over inside iMessage**, built on Photon Spectrum. We designed it against the track's four criteria.

![Coordinate](https://raw.githubusercontent.com/kbhatnagar1506/eyes/main/devpost/gallery/05-coordinate.jpg)

**Vision: hybrid intelligence.** It isn't one chatbot. Eight agents (Sentinel, Network, Dispatch, Police, EMS, Safety, Care, Intake) see each other's lines and work *with* the humans in the thread. The agents page responders and keep the record; the humans decide. Reply "on my way" and the incident turns green. Tap "Person injured" on the officer's phone and Care pages EMS. The people nearby aren't an audience either: they check in, report, and get looked after.

**Craft: a delightful default.** An emergency group chat that buzzes seven times is a group chat people mute. So Eyes is **calm by default**:
- **One live status card per incident, edited in place** as responders confirm and people check in (edits don't buzz).
- The agents' whole brief arrives as **one threaded reply** under it, and police photos thread there too.
- **Native tapbacks instead of noise:** 👍 on "on my way", ❤️ on "I'm safe".
- The Loud effect only where a person's safety is at stake: the area alert and the EMS page.

**Depth: real-world context and edge cases.**
- **Anyone can report.** Text "there's a fight at Woodruff Park": the Intake agent (Gemini, with a keyword fallback if the model is down) geocodes the place and raises a real incident, the same one a camera would. With no place given it asks "Where is this happening?". "Someone collapsed **here**" uses your saved place.
- **A second report** within 400 m joins the live incident instead of duplicating it, and the card shows "reported by 2 people". One new incident per person per 2 minutes.
- **Context that persists across time and channels:** sign-ups and every incident live in Postgres. The same incident is one record across the camera, the mesh, the phone screens, the map and iMessage, so when an officer taps Respond on the phone app, the iMessage card updates.
- **Care, done carefully:** EMS gets a loud page with a map link and the latest camera frame; the person gets standard lay first aid (pressure on bleeding, don't move a possible neck injury) and everything they say is **relayed live to the EMS crew**.

**Traction: trust and safety.** We built the behaviors that earn trust before the flashy ones:
- **Grounded facts only.** No agent may say someone is "on the way" until a human confirms. Care never claims to be physically with anyone.
- **Every danger message says to call 911.** Eyes doesn't pretend to replace 911.
- **Opt-in only.** People text a place to join and STOP to leave, and leaving deletes them. Phone numbers never touch the mesh; the map shows counts and ~100 m dots.
- **Proportionate.** An assault alert says "avoid the area, stay in busy, well-lit places", never active-shooter lockdown language.
- **Tested both ways:** 24 intake checks pass with the model up *and* down, plus the full area-alert and care flow. It ran live to a real phone end to end with zero send errors.

![Protect](https://raw.githubusercontent.com/kbhatnagar1506/eyes/main/devpost/gallery/06-protect.jpg)

### 🛡 Shock Tracker: Zero Data

A safety network sounds like it should know everything about everyone. Eyes was built to prove it doesn't have to.

- **No accounts. No logins. No profiles.** Responder screens join by scanning a QR code. The public side is a text message: no app, no sign-up form.
- **No faces, no identity.** AttackNet reads 17 skeleton points per person. It can't tell who you are, only how bodies are moving, so it works on grainy 320×240 CCTV and has no use for faces.
- **No tracking.** Eyes never follows a person from camera to camera. It knows *that* something is happening and *where*, never *who*.
- **Phone numbers never leave the gateway.** Every device on the mesh can read the incident, so we keep numbers off it entirely; the map shows counts and dots rounded to ~100 m.
- **What we do keep, and why:** the incident record and its evidence frame, because police need it. Alert sign-ups are opt-in, and STOP deletes them.

Useful safety software doesn't need to know who you are. It just needs to know someone needs help.

---

## 🧗 Challenges we ran into

- **Our first model wasn't good enough, and we said so.** v1, trained on 8 clips, scored a 56% F1, barely better than guessing on some clips. Instead of shipping red boxes we couldn't defend, we pretrained on 2,000 real-world clips on a T4, and every number improved on footage the model never saw.
- **The fine-tune that made it worse.** Adapting to our CCTV *looked* right and scored lower on unseen clips, so we threw it away. The hardest engineering call was the one that deleted our own work.
- **iMessage rejected every message.** Photon's shared lines only message registered users who have texted the line first; we found it in the docs, verified it with a listener, and built the "text us first" onboarding around it.
- **Seven texts per incident is a chat people mute.** We rebuilt the iMessage layer around one edited card, threaded replies and tapbacks.
- **Offline alerts are a consistency problem.** Two devices editing the same incident with no server needed hybrid logical clocks and field-level merges, plus chunking so a 40 KB photo never blocks a 200-byte alert.

## 🏅 Accomplishments we're proud of

- **91.5% on 400 clips the model never saw**, from a 1.9M-weight model that reads skeletons, not faces.
- A punch on camera → a red box → the officer's phone → an iMessage card, end to end, live.
- **Our own mesh** that gets the alert out over whatever link still works.
- **A team of AI agents in iMessage that's calm, grounded and opt-in**, built for the people involved, not for a demo reel.

## 📚 What we learned

- **Honest evaluation beats impressive numbers.** The moment we scored only on unseen clips, we knew which work actually mattered.
- **Body language is relational.** A fight lives *between* people, which is why the people transformer exists.
- **In an emergency, the best agent is the quiet one.** Fewer, better messages build more trust than more messages.

## 🚀 What's next

- **Run on real camera feeds** at a pilot site: a campus, a transit station or a parking deck.
- **Connect to real dispatch** (computer-aided dispatch and Rapid SOS-style integrations) instead of demo responder lines.
- **More data on the hard cases:** sparring, horseplay and crowded scenes, where the model still struggles.
- **Real radio:** swap the simulated mesh radio for LoRa or Bluetooth hardware.

---

## 🛠 Built with

`typescript` · `bun` · `vite` · `pytorch` · `yolov8` · `mediapipe` · `densepose` · `detectron2` · `nvidia-t4` · `google-compute-engine` · `vertex-ai` · `gemini` · `photon-spectrum` · `imessage` · `cloud-sql` · `postgresql` · `google-cloud-storage` · `webrtc` · `websockets` · `cloudflare` · `vercel` · `leaflet` · `openstreetmap` · `marta-gtfs` · `chart.js` · `hugging-face`
