for i in training/image_2/*; do
  echo -n "$i "
  echo $i | perl -pe "s|im|gt_im|" | perl -pe "s|_([0-9]+\.png)|_road_\1|"
done
