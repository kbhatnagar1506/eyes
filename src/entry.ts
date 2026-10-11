// ?role=driver | rider | police opens the phone screen; anything else is the
// laptop console (dataset tool, live camera, mesh map).
const role = new URLSearchParams(location.search).get("role");

if (role === "driver" || role === "rider" || role === "police") {
  void import("./phone").then((m) => m.startPhone(role));
} else {
  void import("./main");
}
