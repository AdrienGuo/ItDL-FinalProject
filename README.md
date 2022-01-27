# ItDL-FinalProject
Final Project of my Introduction to Deep Learning Class

DAE.py is a test version.<br>
The main.py is the right code for processing.

## Denoising Autoencoder

Denoising Autoencoder (簡稱 DAE) 和上面的 Sparse Autoencoder 一樣都是 Autoencoder 的其中一種延伸應用。Denoising Autoencoder 是將原始資料加上雜訊後，再經由 Autoencoder 網路去學習變回原始資料；這裡的雜訊可以是 Gaussian Noise 或是使用 Dropout，如圖片；藉由這樣的方法，讓網路能夠學習原始資料中的 “隱含資料” (通常稱為 codings)，並且在將 codings 還原回資料的時候又不會造成 overfitting 的情況 (因為有加上雜訊的關係)。

<a href="https://imgur.com/zso7P1a"><img src="https://i.imgur.com/zso7P1a.png" width="400" title="source: imgur.com" /></a>


