# mCNN-SLC-Membrane
Integrating Pre-Trained Protein Language Model and Multiple Window Scanning Deep Learning Networks for Accurate Identification of Secondary Active Transporters in Membrane Proteins.
We suggest combining multiple-window scanning with pre-trained protein language model encodings in order to identify secondary active transport proteins. Protein sequences contain biological information that we hope to decode. We believe that multiple window scanning and protein-pretrained language model embedding can aid in the comprehension of membrane proteins versus secondary active transporters. We provide efficient identification and classification techniques that aid biologists in understanding these transporters and their intricate role in cancer.

##Methodology
The primary data utilized in this study was obtained from the Universal Protein (UniProt) database [1], specifically focusing on secondary active transporters and membrane proteins. After preprocessing, we used Prottrans, a previously trained protein language model [2], to extract complex features from protein sequences and fine-tune them using our dataset. This method enabled us to capture the encoded information within the sequences, allowing for more accurate predictive modeling. To comprehensively analyze protein sequences, our model architecture is based on convolutional neural networks (CNNs) with numerous window scanning techniques. This strategy considered various window sizes, allowing for a comprehensive examination of sequence patterns. The complete architecture is depicted in Figure 1.



##References
1.	UniProt: the Universal Protein knowledgebase in 2023. Nucleic Acids Research, 2023. 51(D1): p. D523-D531.
2.	Elnaggar, A., et al., ProtTrans: Toward Understanding the Language of Life Through Self-Supervised Learning. IEEE Trans Pattern Anal Mach Intell, 2022. 44(10): p. 7112-7127.
