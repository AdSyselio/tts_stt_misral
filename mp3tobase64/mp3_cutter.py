from pydub import AudioSegment
import argparse
import os


def cut_mp3(input_path: str, start_sec: float, duration_sec: float, output_path: str | None = None):
    """Découpe un fichier MP3.

    Args:
        input_path: Chemin du MP3 source.
        start_sec: Temps de départ en secondes.
        duration_sec: Durée du segment en secondes.
        output_path: Chemin du MP3 de sortie. Si None, crée un nom dérivé.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Le fichier '{input_path}' est introuvable.")

    if start_sec < 0 or duration_sec <= 0:
        raise ValueError("Le temps de départ doit être >= 0 et la durée > 0.")

    # Chargement de l'audio (format déduit automatiquement par pydub/ffmpeg)
    audio = AudioSegment.from_file(input_path)

    end_ms = (start_sec + duration_sec) * 1000
    start_ms = start_sec * 1000

    if end_ms > len(audio):
        raise ValueError(
            f"La fin du segment ({end_ms/1000:.2f}s) dépasse la longueur totale du fichier ({len(audio)/1000:.2f}s)."
        )

    segment = audio[start_ms:end_ms]

    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_segment_{int(start_sec)}s_{int(duration_sec)}s{ext}"

    segment.export(output_path, format="mp3")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Découper un MP3 en spécifiant le temps de départ et la durée.")
    parser.add_argument("--input", "-i", required=True, help="Chemin du fichier MP3 source.")
    parser.add_argument("--start", "-s", type=float, required=True, help="Temps de départ en secondes.")
    parser.add_argument("--duration", "-d", type=float, required=True, help="Durée du segment en secondes.")
    parser.add_argument("--output", "-o", help="Chemin du fichier MP3 de sortie (facultatif).")

    args = parser.parse_args()

    try:
        out = cut_mp3(args.input, args.start, args.duration, args.output)
        print(f"Segment exporté vers : {out}")
    except Exception as e:
        print(f"Erreur : {e}")
        exit(1)


if __name__ == "__main__":
    main() 