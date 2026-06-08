#!/bin/bash

# Data Collection Simulations
# Runs 4 sequential simulations with varying count, weather, ego-speed, and random-seed

set -e  # Exit on any error

COMMON_ARGS="--mode rgb \
  --image-size 1600 900 \
  --towns Town10HD_Opt \
  --traffic dense \
  --scenes-per-combo 1 \
  --max-scenes 2 \
  --root ../data/nuscenes/rgb"

echo "========================================"
echo " Starting 4 Data Collection Simulations"
echo "========================================"

echo ""
echo "[1/4] count=8 | weather=ClearNight | ego-speed=medium | seed=1121458"
python generate.py $COMMON_ARGS \
  --count 8 \
  --weathers ClearNight \
  --ego-speeds medium \
  --random-seed 1121458

echo ""
echo "[2/4] count=10 | weather=HardRainNoon | ego-speed=fast | seed=3847291"
python generate.py $COMMON_ARGS \
  --count 10 \
  --weathers HardRainNoon \
  --ego-speeds fast \
  --random-seed 3847291

echo ""
echo "[3/4] count=12 | weather=FoggyMorning | ego-speed=slow | seed=7563920"
python generate.py $COMMON_ARGS \
  --count 12 \
  --weathers FoggyMorning \
  --ego-speeds slow \
  --random-seed 7563920

echo ""
echo "[4/4] count=14 | weather=CloudyNoon | ego-speed=medium | seed=2948765"
python generate.py $COMMON_ARGS \
  --count 14 \
  --weathers CloudyNoon \
  --ego-speeds medium \
  --random-seed 2948765

echo ""
echo "========================================"
echo " All 4 simulations completed."
echo "========================================"
