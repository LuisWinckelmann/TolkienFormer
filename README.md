<!-- README.md -->
<!-- Project Top -->
<a name="readme-top"></a>

<h1 align="center">TolkienFormer - Textgeneration with Tolkiens Touch</h1>
  <p align="center">
    A project exploring LSTM and Transformer-like models with with implementations in Python & Pytorch.
  <br />
  <a href="#results"><strong>Example Results »</strong></a>
</p>


<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#about-the-project">About The Project</a></li>
    <li><a href="#setup">Setup</a></li>
    <li><a href="#data">Data</a></li>
    <li><a href="#training">Training</a></li>
    <li><a href="#testing">Testing</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#results">Results</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>


<!-- ABOUT THE PROJECT -->
## About The Project
TolkienFormer is a project dedicated to get more familiar with Generative Neural Networks and Pytorch. The codebase contains 

TODO: Write description <br>
TODO: Short showcase

## Setup 
  First, you have to clone the repo and create a conda environment, as well adding the project root to your PYTHONPATH to enable local imports:
   ```shell
   # 1. Clone this repository
   git clone https://github.com/LuisWinckelmann/TolkienFormer.git
   cd TolkienFormer
   # 2. Setup conda env
   conda create --name tolkienformer
   conda activate tolkienformer
   conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
   # 3. Enable local imports by adding the root to your pythonpath:
   # 3a) Linux:
   export PYTHONPATH=.:$PYTHONPATH
   # 3b) Windows:
   set PYTHONPATH=%PYTHONPATH%;%cd%
   ```

## Data
  For the data you can use any *.txt file that you want. In the current setup the file will get parse row-wise.
  The example dataset chapter1, provided in `src/data/chapter1` includes chapter 1 or Tolkien's The Lord of the rings obtained from [here](https://ae-lib.org.ua/texts-c/tolkien__the_lord_of_the_rings_3__en.htm).
  To use your own dataset simply copy the text file(s) into `src/data` and run:
  ```shell
  cd src/data
  python data_preparation.py 
  ```
  If your data is stored somewhere else then `src/data` you can use `--path_to_folder_with_txt_filess` to adjust the root folder with the .txt files inside.
  If your data has another format you'll need to adjust your custom dataset in `src/utils/datasets.py` accordingly.

## Training
  To run training of the LSTM run:
  ```shell
  cd src/models/lstm
  python train.py 
  ```
  To run training of the transformer-like model run:
  ```shell
  cd src/models/transformer
  python train.py 
  ```
  All currently available hyperparameters can be changed in the corresponding config.json files located in `src/modes/lstm` or `src/modes/transformers` respectively. 

## Testing
  After executing the training, to generate results of the models as shown in the <a href="#about-the-project">description</a>, you can run:
  ```shell
  # LSTM model
  cd src/models/lstm
  python test.py 
  # Transformer-like model
  cd src/models/transformer
  python test.py 
  ```
  The parameters for the evaluation can be changed in the model `config.json`.

## Roadmap
- [X] Confirm setup written in README works on a new machine
- [X] Docstrings
- [X] Setting Flags instead of hardcoded Parameters like NUM_PREDICTED_SENTENCES and LOADING_MODEL_EPOCH and DATA_PATH
- [X] Use Typing
- [ ] Move to logging from printing
- [ ] Write description with a showcase
- [ ] Publish some additional results
- [ ] Get rid of code doubling my merging LSTM & Transformer folders and specifically train.py & test.py 

## Results
TODO Write summary of results and link to some report that i maybe also upload?!

## License
Distributed under the MIT License. See `LICENSE.txt` for more information.

## Contact
[![LinkedIn][linkedin-shield]][linkedin-url] <br>
Luis Winckelmann  - luis.winckelmann@gmx.com <br>
Project Link: [https://github.com/LuisWinckelmann/project_name](https://github.com/LuisWinckelmann/JavaDeep-MLP-RNN-from-scratch-in-Java)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

[license-shield]: https://img.shields.io/github/license/LuisWinckelmann/JavaDeep-MLP-RNN-from-scratch-in-Java.svg?style=for-the-badge
[license-url]: https://github.com/LuisWinckelmann/JavaDeep-MLP-RNN-from-scratch-in-Java/blob/main/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/luiswinckelmann
[PyTorch]: https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white
