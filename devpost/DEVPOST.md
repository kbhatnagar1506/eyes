# 👁 Eyes: cameras that watch. Eyes that act.

> **Every city is already covered in cameras. Not one of them has ever called for help.**
> Eyes turns the cameras we already have into a safety network that **detects** danger, **dispatches** help and **warns the people nearby**, even when the internet is down, coordinated by AI agents that live in iMessage.

![Eyes](https://raw.githubusercontent.com/kbhatnagar1506/eyes/main/devpost/gallery/01-eyes.jpg)

**Live app:** https://eyes-hacklanta.vercel.app (Dataset tool · Live demo · Mesh · Model) · **Code:** https://github.com/kbhatnagar1506/eyes · **iMessage agents:** come to our table and text the Eyes line

---

## 💡 Inspiration

Imagine your sister walking home after dark.

A man starts following her. She speeds up; so does he. She crosses the street; he crosses too. She runs, and he runs after her. She passes a bus stop, a store, a parking deck, and at every one of them a camera turns its glass eye toward her and **watches**. It records every second of the worst night of her life.

And it does nothing. It doesn't call anyone. It doesn't warn the people a block away. It doesn't tell the bus driver pulling up to the stop. Tomorrow someone might pull the footage; tonight the camera is just a witness that never speaks.

**The infrastructure to protect her already exists. It's bolted to every wall she ran past.** It just has no brain and no voice. So we gave it both.

---

## ⚡ What it does

| | Layer | What happens |
|---|---|---|
| 👁 | **Detect** | Every camera frame goes to **AttackNet, our own model, running live on a Google Cloud NVIDIA T4**. It reads the body language of everyone in frame, and the moment a scene turns violent a red box lands on the person attacking. The evidence frame is captured that instant. |
| 🕸 | **Spread** | The incident spreads across a **mesh of 130 city nodes** (police, hospitals, MARTA stations, live buses) over whatever link works: internet, P2P Wi-Fi or mesh radio. It gets out **even offline**. |
| 💬 | **Coordinate** | A team of AI agents takes over in **iMessage**: pages the nearest responders, sends police the camera frame, tracks who is actually on the way, and keeps one calm live status card updated. |
| 🛡 | **Protect** | Anyone within 800 m who signed up gets an alert ("⚠️ be careful around Woodruff Park"), a safe spot and an "Are you safe?" poll. Anyone hurt gets first aid by text until EMS arrives. **Anyone can report** an incident just by texting what's happening and where. |

The camera stops being a witness and becomes a first responder.

![Response flow](https://raw.githubusercontent.com/kbhatnagar1506/eyes/main/devpost/gallery/13-flow.jpg)

---

## 🧠 How it works

### AttackNet: our model, live on a GPU

Pixels are a bad way to understand a fight: they carry lighting, clothing and faces, and none of that is what violence *is*. Violence is **how bodies move relative to each other**. So AttackNet never sees pixels; it sees **skeletons**.

- **YOLOv8x-pose** finds every person and 17 keypoints, and we turn each person into **84 body-language signals** per frame: posture and lean, someone down, limb speed, arms up shielding the head, closing in, contact, the crowd converging or scattering.
- A **temporal transformer** reads each person's last two seconds, because danger is a sequence: tension builds, someone lunges, someone recoils.
- A **people transformer** lets everyone in the frame attend to everyone else, because an attacker, a victim and a bystander can look identical alone.
- **1,899,209 weights**, small on purpose: for 2,000 training clips, a bigger model memorizes videos instead of learning patterns.

**The labeling trick.** Datasets say *when* a fight happens, never *who* is fighting. We train with multiple-instance learning: a frame's score is its most violent person, a clip's score is its most violent moments, and the per-person scores learn *who* is involved on their own. That's why the red box lands on the person throwing the punch, live and on the CCTV clip below, though no one ever told the model which one it was.

![Detect](https://raw.githubusercontent.com/kbhatnagar1506/eyes/main/devpost/gallery/02-detect.jpg)

### Trained on a Google Cloud NVIDIA T4, scored honestly

We pretrained on **Real Life Violence Situations** (1,000 fight + 1,000 normal real-world clips, split 70/10/20 by clip), then tested on footage the model **never trained on**: 400 held-out clips, our own hand-labeled CCTV scored one clip at a time, and sparring clips kept out entirely. YOLOv8x-pose followed 29,869 people across 98,000 frames on the T4; the VM is created per run and deleted after.

| Test (never trained on) | v1 (our clips only) | **v2 (pretrained on RLVS)** |
|---|---|---|
| 400 held-out real-world clips | not tested | **91.5% accuracy · AUC 0.975** |
| Our hand-labeled CCTV | F1 56% · AUC 0.64 | **F1 63% · AUC 0.72** |
| Showcase CCTV assault clip | | **F1 83%** |
| Sparring | | F1 61% · AUC 0.51 |

We also tried fine-tuning on our CCTV. It scored *worse* on unseen clips, so we didn't ship it, and the Model page says so. Sparring is still hard, because it genuinely looks like fighting, and we show that too.

![Results](https://raw.githubusercontent.com/kbhatnagar1506/eyes/main/devpost/gallery/09-results.jpg)

### Running it live

The live demo runs the **exact pipeline AttackNet was trained on**, not a lighter copy. The camera streams frames at 10 fps (the rate the model learned at) over a WebSocket to a **Google Cloud NVIDIA T4**: YOLOv8x-pose, our tracker, the same 84 body-language features, then AttackNet, in **about 45 ms per frame on the GPU and about 90 ms round trip**. A box turns red above 30%, and the alert fires when the scene stays above 50% for two frames in a row.

We tested it on a recording of the two of us it had never seen: it caught 3 of 5 punches about a second after each began (it misses short jabs when we're both cut off at the head), with **zero false alarms** on calm footage of people talking and waving at the camera. Live, with full swings in frame, it fires on every punch we've thrown at it.

A second detector stands by: **StrikeDetector** runs entirely in the browser (MediaPipe pose and a geometric hit model that fires when a hand enters another person's body faster than three torso-lengths per second). If the GPU is ever unreachable, it takes over, so the camera never goes blind.

### The mesh

The alert has to get out even when the internet doesn't, so we built our own replicated store: hybrid logical clocks with field-level last-writer-wins, so every device can write and all of them converge. Fresh changes gossip in 48 KB chunks, so a 40 KB photo never blocks the 200-byte alert ahead of it. A networking agent measures every link and sends each message over the best one, with the reason. The Atlanta mesh has 130 city nodes, 43 live MARTA buses, 2,155 public places and 7,011 stops.

![Mesh](https://raw.githubusercontent.com/kbhatnagar1506/eyes/main/devpost/gallery/03-spread.jpg)

---

## 🏗 Architecture

![Infrastructure](https://raw.githubusercontent.com/kbhatnagar1506/eyes/main/devpost/gallery/12-infrastructure.jpg)

| Layer | Built with |
|---|---|
| Edge | Camera streams to AttackNet on the T4 · MediaPipe + StrikeDetector in the browser as the fallback · responder screens join by QR |
| Mesh | Our own HLC + LWW store · WebRTC P2P · simulated mesh radio · Bun hub behind a Cloudflare tunnel |
| Agents | Bun · Gemini 2.5 Flash on Vertex AI · Photon Spectrum (iMessage) |
| Model | YOLOv8x-pose · PyTorch AttackNet, trained **and served live** on an NVIDIA T4 (Compute Engine) · DensePose renders |
| Data | Cloud SQL Postgres · Cloud Storage (private, signed links) |
| Web | Vite + TypeScript · Leaflet + OpenStreetMap · Chart.js · Vercel |

---

## 🏆 Sponsor tracks

### 🚗 Mercedes-Benz: Best Application of Computer Vision

Computer vision that **acts**. **AttackNet**, our own 1.9M-weight model, reads body language from skeletons, scores **91.5% / AUC 0.975 on 400 real-world clips it never saw** and 83% F1 on unseen CCTV, and **runs live on a Google Cloud NVIDIA T4** at about 45 ms a frame, the same pipeline it was trained on. It puts the red box on the attacker, not just the scene, stays green on calm footage, and its detection is what dispatches police: the frame that turns red is the frame an officer receives on their phone. Around it: **StrikeDetector** in the browser as an always-on fallback, and **DensePose** on the T4 mapping every visible body for the Dataset tool. Privacy-first throughout: 17 skeleton points, never faces.

### 💬 Photon: Agents in iMessage

When the camera sees something, **a team of AI agents takes over inside iMessage**, built on Photon Spectrum.

![Coordinate](https://raw.githubusercontent.com/kbhatnagar1506/eyes/main/devpost/gallery/05-coordinate.jpg)

- **Vision: hybrid intelligence.** Eight agents (Sentinel, Intake, Network, Dispatch, Police, EMS, Safety, Care) see each other's lines and work *with* the humans in the thread. Agents page and keep the record; humans decide. Reply "on my way" and the incident turns green; tap "Person injured" on the officer's phone and Care pages EMS.
- **Craft: calm by default.** An emergency chat that buzzes seven times gets muted. So there's **one live status card per incident, edited in place** (edits don't buzz), the agents' brief arrives as **one threaded reply**, and acknowledgements are **native tapbacks**: 👍 for "on my way", ❤️ for "I'm safe". The Loud effect is saved for the area alert and the EMS page.
- **Depth: real-world context.** Text "there's a fight at Woodruff Park" and the Intake agent finds the place and raises a real incident; no place, and it asks where; "someone collapsed *here*" uses your saved place. A second report nearby joins the live incident. Care relays everything the person says to the EMS crew. One incident is one record across the camera, the mesh, the phones, the map and iMessage.
- **Traction: trust.** No agent says someone is "on the way" until a human confirms. Every report and care message tells people to call 911. Alerts are opt-in (text a place to join, STOP to leave and be deleted) and proportionate ("avoid the area", never lockdown language). 24 intake checks pass with the model up *and* down, and it ran live to a real phone end to end.

![Protect](https://raw.githubusercontent.com/kbhatnagar1506/eyes/main/devpost/gallery/06-protect.jpg)

### 🛡 Shock Tracker: Zero Data

A safety network sounds like it should know everything about everyone. Eyes proves it doesn't have to: **no accounts, no app, no profiles** (responders join by QR; the public side is a text message). **No faces and no identity**: AttackNet reads 17 skeleton points, so it can't tell who you are, only how bodies move. **No tracking** from camera to camera. **Phone numbers never leave the gateway**; the shared map shows counts and dots rounded to about 100 m. We keep only the incident record and its evidence frame, because police need them. Safety software doesn't need to know who you are. It just needs to know someone needs help.

---

## 🧗 Challenges

- **Our first model wasn't good enough, and we said so.** v1 scored 56% F1. Instead of shipping red boxes we couldn't defend, we pretrained on 2,000 real-world clips, and every unseen-footage number went up.
- **The fine-tune that made it worse.** It looked right and scored lower on unseen clips, so we deleted it. The hardest call was throwing away our own work.
- **iMessage rejected every message** until we found that Photon's shared lines only reach registered users who've texted first, and built onboarding around it.

## 🏅 Accomplishments

- **91.5% on 400 clips the model never saw**, from a 1.9M-weight model that reads skeletons, not faces.
- **Our own model, live:** a punch on camera → AttackNet on a GPU in the cloud → a red box on the attacker → the officer's phone → an iMessage card, end to end, in seconds.
- Our own offline-first mesh, and AI agents in iMessage that are calm, grounded and opt-in.

## 📚 What we learned

- **Honest evaluation beats impressive numbers.**
- **A fight lives between people**, which is why the people transformer exists.
- **In an emergency, the best agent is the quiet one.**

## 🚀 What's next

- **AttackNet on real camera feeds** at a pilot site: a campus, a transit station or a parking deck, with one GPU serving many cameras.
- **Real dispatch integration** instead of demo responder lines.
- **More hard cases** (sparring, horseplay, crowds) and **real radio** (LoRa or Bluetooth) in place of the simulated mesh radio.
