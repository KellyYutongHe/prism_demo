# Demo code for ECCV Submission \#8438 "Automated Black-box Prompt Engineering for Personalized Text-to-Image Generation"

To perform subject-drive personalized T2I generation (e.g. for DreamBooth objects), run
```
python main_obj_invert.py --obj OBJ_NAME --n-streams NUM_STREAMS --n-iterations NUM_ITERATIONS --goal_dir DATASET_DIR
```

To perform direct image inversion, run
```
python main_img_invert.py --obj IMG_FILE_NAME --n-streams NUM_STREAMS --n-iterations NUM_ITERATIONS --goal_dir DATASET_DIR
```

To perform style-drive personalized T2I generation (e.g. for WikiArt style), run
```
python main_style_invert.py --obj STYLE_NAME --n-streams NUM_STREAMS --n-iterations NUM_ITERATIONS --goal_dir DATASET_DIR
```
