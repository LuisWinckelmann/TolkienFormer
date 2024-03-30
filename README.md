<a name="readme-top"></a>

<br />
<div align="center">
<!-- PROJECT LOGO 
  <a href="https://github.com/othneildrew/Best-README-Template">
    <img src="images/logo.png" alt="Logo" width="80" height="80">
  </a>
 -->
  <h3 align="center">WIP - WORK in Progress</h3>
  <p align="center">
    TODO: Description
    <br />
    <a href="https://github.com/othneildrew/Best-README-Template"><strong>Example Results »</strong></a>
    <br />
    <br />
  </p>
</div>


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
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>


<!-- ABOUT THE PROJECT -->
## About The Project

TODO: Short description

TODO: Short showcase

## Setup 
  First, you have to clone the repo and create a conda environment.
   ```sh
   # 1. Clone this repository
   git clone https://github.com/LuisWinckelmann/TolkienFormer.git
   cd TolkienFormer
   # 2. Setup python env
   chmod +x setup_env.sh
   ./setup_env.sh
   
   conda activate TolkienFormer
   ```

## Data
  For the data you can use any *.txt file that you want. In the current setup the file will get parse row-wise.
  The example dataset chapter1, provided in `src/data/chapter1` includes chapter 1 or Tolkien's The Lord of the rings obtained from [here](https://ae-lib.org.ua/texts-c/tolkien__the_lord_of_the_rings_3__en.htm).
  To use your own dataset simply copy the text file(s) into `src/data` and run `./prepare_data.sh`. If your data has another format you'll need to adjust your custom dataset in `src/utils/datasets.py` accordingly.

## Training
  To run training of the LSTM on the provided dataset, run:
  ```
  python src/modules/lstm/train.py 
  ```
  To run training of the transformer-like model on the provided dataset, run:
  ```
  python src/modules/transformer/train.py 
  ```
## Testing
  After executing the training, to generate results of the models as shown in the <a href="#about-the-project">description</a>, you can run:
  ```
  python src/modules/lstm/test.py 
  ```
  ```
  python src/modules/transformer/test.py 
  ```
  The parameters for the evaluation can be changed in the model `config.json`.
<!
## Roadmap
- [ ] Push working configurations
- [ ] Create shell scripts for example usage
- [ ] Setup & Dependencies
- [ ] Write README
- [ ] Confirm cloning & following README works
- [ ] Leftover code beautification & Bugfixes
  - [ ] Get rid of code doubling my merging train & test
  - [ ] Move to logging from printing
  - [ ] Enable GPU as a device, currently buggy sometimes
  - [ ] Setting Flags instead of hardcoded Parameters like NUM_PREDICTED_SENTENCES and LOADING_MODEL_EPOCH
  - [ ] Use Typing
-->

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