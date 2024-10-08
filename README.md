# mCNN-SAT-Membrane
Integrating Pre-Trained Protein Language Model and Multiple Window Scanning Deep Learning Networks for Accurate Identification of Secondary Active Transporters in Membrane Proteins.
We suggest combining multiple-window scanning with pre-trained protein language model encodings in order to identify secondary active transport proteins. Protein sequences contain biological information that we hope to decode. We believe that multiple window scanning and protein-pretrained language model embedding can aid in the comprehension of membrane proteins versus secondary active transporters. We provide efficient identification and classification techniques that aid biologists in understanding these transporters and their intricate role in cancer.

## Fig. 1: A comprehensive representation of the overall framework employed in the proposed study
![](https://github.com/Malik-glt/mCNN-SLC-Membrane/blob/main/Architecture%20of%20Project.png?raw=true)

## Methodology
The primary data utilized in this study was obtained from the Universal Protein (UniProt) database [1], specifically focusing on secondary active transporters and membrane proteins. After preprocessing, we used Prottrans, a previously trained protein language model [2], to extract complex features from protein sequences and fine-tune them using our dataset. This method enabled us to capture the encoded information within the sequences, allowing for more accurate predictive modeling. To comprehensively analyze protein sequences, our model architecture is based on convolutional neural networks (CNNs) with numerous window scanning techniques. This strategy considered various window sizes, allowing for a comprehensive examination of sequence patterns. The complete architecture is depicted in Figure 1 and the classification model in Figure 2.

## Fig. 2: Deep Learning Model: Multiple Windows Scanning mCNN

![](https://github.com/Malik-glt/mCNN-SLC-Membrane/blob/main/mCNN%20Model.png?raw=true)

## References
1.	UniProt: the Universal Protein knowledgebase in 2023. Nucleic Acids Research, 2023. 51(D1): p. D523-D531.
2.	Elnaggar, A., et al., ProtTrans: Toward Understanding the Language of Life Through Self-Supervised Learning. IEEE Trans Pattern Anal Mach Intell, 2022. 44(10): p. 7112-7127.

## Citation

If you use this repository, please cite the following paper:

```bibtex
@ARTICLE{malik2023integrating,
  author = {Muhammad Shahid Malik and Yu-Yen Ou},
  title = {Integrating Pre-Trained protein language model and multiple window scanning deep learning networks for accurate identification of secondary active transporters in membrane proteins},
  journal = {Methods},
  volume = {220},
  pages = {11--20},
  year = {2023},
  publisher = {Elsevier},
  doi = {10.1016/j.ymeth.2023.10.008}
}
