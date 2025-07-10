import argparse
import os
import shlex
import subprocess
from pathlib import Path

# Chemin FFmpeg par défaut. Peut être redéfini par l'argument CLI ou la variable d'env FFMPEG_PATH
DEFAULT_FFMPEG = os.getenv("FFMPEG_PATH", "ffmpeg")


def ffmpeg_installed(ffmpeg_cmd: str) -> bool:
    """Vérifie que la commande ffmpeg est disponible."""
    try:
        subprocess.run([ffmpeg_cmd, "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def cut_audio(input_path: str, start_sec: float, duration_sec: float, output_path: str | None = None, reencode: bool = False, ffmpeg_cmd: str = DEFAULT_FFMPEG):
    """Découpe n'importe quel fichier audio supporté par FFmpeg.

    Si reencode=False, on utilise une copie directe (-c copy) qui est très rapide, mais
    certaines combinaisons conteneur/codec peuvent refuser la copie. Dans ce cas,
    passez reencode=True pour réencoder en WAV (pcm_s16le) ou MP3 (libmp3lame).
    """
    if not ffmpeg_installed(ffmpeg_cmd):
        raise RuntimeError("FFmpeg n'est pas installé ou pas présent dans le PATH.")

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Fichier introuvable : {input_path}")

    if start_sec < 0 or duration_sec <= 0:
        raise ValueError("Le temps de départ doit être >=0 et la durée >0")

    input_path = os.path.abspath(input_path)
    if output_path is None:
        p = Path(input_path)
        output_path = str(p.with_stem(f"{p.stem}_segment_{int(start_sec)}s_{int(duration_sec)}s"))

    # Commande FFmpeg
    cmd = [
        ffmpeg_cmd,
        "-y",  # overwrite sans demander
        "-ss",
        str(start_sec),
        "-t",
        str(duration_sec),
        "-i",
        input_path,
    ]

    if reencode:
        # Choisit un codec en fonction de l'extension de sortie
        ext = Path(output_path).suffix.lower()
        if ext in {".wav", ".wave"}:
            cmd += ["-acodec", "pcm_s16le"]
        elif ext in {".mp3"}:
            cmd += ["-acodec", "libmp3lame"]
        else:
            cmd += ["-acodec", "copy"]  # tentative copie si codec inconnu
    else:
        cmd += ["-c", "copy"]

    cmd.append(output_path)

    # Exécution
    print("Exécution :", " ".join(shlex.quote(str(c)) for c in cmd))
    process = subprocess.run(cmd, capture_output=True, text=True)
    if process.returncode != 0:
        raise RuntimeError(f"FFmpeg a échoué :\n{process.stderr}")

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Découper un fichier audio via FFmpeg (sans pydub).")
    parser.add_argument("--input", "-i", required=True, help="Chemin du fichier source.")
    parser.add_argument("--start", "-s", type=float, required=True, help="Temps de départ en secondes.")
    parser.add_argument("--duration", "-d", type=float, required=True, help="Durée du segment en secondes.")
    parser.add_argument("--output", "-o", help="Fichier de sortie (sinon auto-généré).")
    parser.add_argument("--reencode", action="store_true", help="Réencoder au lieu de copier le flux audio.")
    parser.add_argument("--ffmpeg", dest="ffmpeg_path", default=DEFAULT_FFMPEG, help="Chemin vers l'exécutable FFmpeg si non présent dans le PATH.")

    args = parser.parse_args()

    try:
        out = cut_audio(args.input, args.start, args.duration, args.output, args.reencode, args.ffmpeg_path)
        print(f"Segment exporté dans : {out}")
    except Exception as exc:
        print(f"Error: {exc}")
        exit(1)


if __name__ == "__main__":
    main() 