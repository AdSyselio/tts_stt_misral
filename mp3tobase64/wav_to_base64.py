import argparse
import base64
import os
from pathlib import Path


def wav_to_base64(input_path: str, as_data_uri: bool = False, wrap: bool = False) -> str:
    """Encode un fichier WAV en chaîne base64.

    Args:
        input_path: Chemin du fichier WAV (ou autre fichier binaire).
        as_data_uri: Si True, préfixe par `data:audio/wav;base64,`.
        wrap: Si True, ajoute une nouvelle ligne tous les 76 caractères (RFC 2045).

    Returns:
        Chaîne base64 (UTF-8).
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Fichier introuvable : {input_path}")

    with open(input_path, "rb") as f:
        raw_bytes = f.read()

    encoded_bytes = base64.b64encode(raw_bytes)
    encoded_str = encoded_bytes.decode("utf-8")

    if wrap:
        import textwrap
        encoded_str = "\n".join(textwrap.wrap(encoded_str, 76))

    if as_data_uri:
        encoded_str = f"data:audio/wav;base64,{encoded_str}"

    return encoded_str


def main():
    parser = argparse.ArgumentParser(description="Encoder un fichier WAV en base64.")
    parser.add_argument("--input", "-i", required=True, help="Chemin du fichier WAV (ou autre) à encoder.")
    parser.add_argument("--output", "-o", help="Fichier où écrire la chaîne base64 (sinon impression stdout).")
    parser.add_argument("--data-uri", action="store_true", help="Préfixer par data:audio/wav;base64,.")
    parser.add_argument("--wrap", action="store_true", help="Retour à la ligne tous les 76 caractères.")

    args = parser.parse_args()

    try:
        b64 = wav_to_base64(args.input, as_data_uri=args.data_uri, wrap=args.wrap)
        if args.output:
            Path(args.output).write_text(b64, encoding="utf-8")
            print(f"Base64 écrit dans : {args.output}")
        else:
            print(b64)
    except Exception as exc:
        print(f"Erreur : {exc}")
        exit(1)


if __name__ == "__main__":
    main() 