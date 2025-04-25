# Code for "Automated Black-box Prompt Engineering for Personalized Text-to-Image Generation"


## Installation
```
conda env create -f environment.yaml
conda activate prism
```

## Specify your OpenAI API key
```
export OPENAI_API_KEY=YOUR-OPENAI-API-KEY
```
You can also add this line in your bashrc file.

## Running experiments
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

## Citation
If you find our work interesting or helpful, please consider citing

```
@article{he2024automated,
  title={Automated Black-box Prompt Engineering for Personalized Text-to-Image Generation},
  author={He, Yutong and Robey, Alexander and Murata, Naoki and Jiang, Yiding and Williams, Joshua Nathaniel and Pappas, George J and Hassani, Hamed and Mitsufuji, Yuki and Salakhutdinov, Ruslan and Kolter, J Zico},
  journal={arXiv preprint arXiv:2403.19103},
  year={2024}
}
```
