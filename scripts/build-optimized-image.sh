#!/bin/bash
set -e

echo "🔨 Construction d'une image optimisée avec cache..."

# Construire l'image avec tous les modèles pré-téléchargés
docker build \
  --build-arg BUILDKIT_INLINE_CACHE=1 \
  --cache-from ghcr.io/adsyselio/tts_stt_misral:latest \
  -t ghcr.io/adsyselio/tts_stt_misral:latest \
  .

echo "📤 Push de l'image optimisée..."
docker push ghcr.io/adsyselio/tts_stt_misral:latest

echo "✅ Image optimisée construite et publiée !"
echo "🚀 Prochain déploiement sera beaucoup plus rapide !" 