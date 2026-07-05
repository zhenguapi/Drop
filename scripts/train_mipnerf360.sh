exp_name='mip12'
# scenes=("bicycle" "bonsai" "counter" "garden" "kitchen" "room" "stump") 
scenes=("bicycle" "bonsai" "counter" "garden") # 12 views 
dataset_path='/root/autodl-tmp/Mip-NeRF360/360_v2/'
n_views=12

for scene in "${scenes[@]}"
do
  echo "Training on $scene..."
  python train.py -s $dataset_path/$scene/ \
    -m output/$exp_name/$scene \
    --eval -r 8 \
    --n_views $n_views

  echo "Rendering $scene..."
  python render.py -m output/$exp_name/$scene -r 8
done

# Compute metrics for all scenes
python metric.py --path output/$exp_name
