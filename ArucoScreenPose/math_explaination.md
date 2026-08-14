# Eye Tracking & Posizionamento 3D

## 1. Sistemi di Riferimento

### ArucoScreenPose (Coordinate Schermo)

- **Origine:** Centro della finestra identificata dai marker ArUco.
- **Unità di misura:** Millimetri (mm) sul piano del monitor (indipendente dalla risoluzione video).
- **Assi:**
  - **X:** Cresce verso destra.
  - **Y:** Cresce verso il basso.
  - **Z:** Cresce allontanandosi dall'osservatore (verso/oltre il monitor).
- **Piani notevoli:**
  - Z = 0: Piano della superficie del monitor.
  - Z < 0: Spazio occupato dall'osservatore.

> **Esempio:** Un marker situato 250 mm a sinistra del centro e 150 mm sopra l'origine avrà coordinate (-250, -150, 0).

---

### Coordinate Camera

- **Origine:** Centro ottico della lente.
- **Unità di misura:** Pixel per lo spazio immagine 2D / Millimetri per lo spazio 3D.
- **Assi:**
  - **X:** Cresce verso destra.
  - **Y:** Cresce verso il basso.
  - **Z:** Cresce procedendo verso lo schermo.
  - Z = 0: Piano della lente della fotocamera.

---

## 2. Risoluzione della Posa (`solvePnP`)

Algoritmo utilizzato per calcolare la posizione e l'orientamento 3D della fotocamera rispetto a un oggetto a geometria nota (i marker sul monitor).

### Output di `solvePnP`

- **R (Rotazione):** Matrice di rotazione 3x3 che trasforma un vettore dal sistema dello **schermo** al sistema della **camera**.
- **t (Traslazione):** Vettore di traslazione in mm.

> **Nota:** Tutte le altre metriche (yaw, pitch, distanza, ecc.) sono derivazioni matematiche dirette di R e t. La matrice trasposta R.T rappresenta la rotazione inversa (da camera a schermo).

---

## 3. Calcoli Geometrici

### Posizione della Camera rispetto allo Schermo (C)

Nel sistema di riferimento della camera, l'origine è (0, 0, 0). Per calcolare le coordinate del centro della camera rispetto allo schermo (C), invertiamo la trasformazione affine:

`P_camera = R x P_schermo + t`

Ponendo la posizione della camera nell'origine del proprio sistema (P_camera = 0):

0 = R x C + t

t = -R x C

C = -R^-1 x t

Poiché R è una matrice ortogonale, l'inversa corrisponde alla trasposta (R^-1 = R.T):

C = -R.T x t

Da C si legge la distanza dal piano dello schermo:

dist = |C_z|

- È la distanza **perpendicolare al vetro**, non la linea d’aria dal centro (quella sarebbe ||C||).
- Il valore assoluto rende positivo C_z (negativo per convenzione: osservatore a Z < 0).

### Orientamento della Camera (look)

Definisce la direzione verso cui punta la lente della fotocamera, espressa nel sistema di coordinate dello schermo.

1. **Vettore di puntamento nativo (Camera):** v_camera = (0, 0, 1) (asse Z entrante/avanti).
2. **Trasformazione nello schermo:**

look = R.T x (0, 0, 1)

#### Rilevanza Pratica

La posizione C descrive _dove si trova_ la camera, mentre look descrive _dove sta puntando_.

- Un dispositivo può trovarsi a 57 cm dallo schermo (C) e puntare al centro, in basso verso la barra delle applicazioni o verso l'esterno.
- Le metriche di **Yaw**, **Pitch** e l'**angolo di incidenza** derivano esclusivamente dal vettore look.

### Calcolo Yaw

yaw = atan2(look_x, look_z) -> poi convertito in gradi.

- `atan2` calcola l'angolo formato dalla coppia di coordinate (look_x, look_z).
- **Interpretazione:** Se vado `look_z` in avanti e `look_x` a destra, di quanti gradi sono girato?
- Non è una misura nuova: è semplicemente la freccia `look` letta lungo la sola componente orizzontale.
- **Segno:** 0° = dritto; positivo = verso destra dello schermo; negativo = verso sinistra.

### Calcolo Pitch

Stessa idea dello yaw, ma sul piano verticale (su/giù) invece che sinistra/destra.

pitch = atan2(-look_y, hypot(look_x, look_z)) -> gradi.

- Si ignora la sola componente “quanto a destra”: si guarda quanto la freccia pende in alto o in basso.
- Y sullo schermo cresce verso il basso, quindi si usa `-look_y`: look_y negativo (punti in alto) → pitch positivo.
- `hypot(look_x, look_z)` è la lunghezza della freccia vista dall’alto (quanto “avanti” resta dopo aver tolto il su/giù). Così il pitch è l’elevazione vera, non una scorciatoia 2D.
- **Segno:** 0° = né su né giù; positivo = verso l’alto; negativo = verso il basso.

### Calcolo Roll

Il roll non dice *dove* punti (quello è look / yaw / pitch), ma **come è ruotata la camera attorno a quell’asse**: tilt orario/antiorario, come inclinare la testa sulla spalla.

1. `cam_up = R.T × (0, -1, 0)` — “su” della camera, detto in coordinate schermo (`-Y` camera = su).
2. Si costruisce un riferimento orizzontale rispetto a look: `plane_right` nel piano XZ, poi `plane_up = plane_right × look`.
3. roll = atan2(cam_up · plane_right, cam_up · plane_up) -> gradi.

- **Segno:** 0° = testa dritta; positivo = orario visto da dietro la camera.

### Angolo di incidenza

Un solo numero al posto di yaw+pitch: quanto look è storto rispetto alla normale dello schermo (asse +Z).

incidenza = acos(clip(look · (0, 0, 1), -1, 1)) -> gradi.

- 0° = inquadratura frontale (look parallelo a +Z).
- `clip` evita che arrotondamenti diano un prodotto scalare > 1 e facciano fallire `acos`.
- Per angoli piccoli: incidenza ≈ √(yaw² + pitch²).

### Errore di riproiezione (riproj)

Controllo di qualità su R e t: si riproiettano i corner 3D dei marker nel frame (`projectPoints`) e si confrontano con quelli rilevati da ArUco.

riproj = media di ||pixel_proiettato − pixel_rilevato||  (pixel)

- Più basso = pose più coerente. Nel codice si scarta il frame se riproj > 12 px.
