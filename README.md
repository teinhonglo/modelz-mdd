# [WIP] End-to-End Mispronunciation Detection and Diagnosis (MDD) Service

This project implements an **End-to-End Mispronunciation Detection and Diagnosis (E2E-MDD)** service based on an **end-to-end phone recognizer** using **wav2vec CTC**. The goal is to provide:

- **MDD Feedback**: Detailed diagnostic feedback on mispronunciations (Dictate).
- **Pronunciation Scoring**: Using **Goodness of Pronunciation (GOP)** to assess phoneme-level scores.
- **Teaching Suggestions**: Automatically generated pedagogical feedback using **Large Language Models (LLMs)**.

## **Features**

- **End-to-End Design**: Streamlines the MDD process by integrating **wav2vec CTC** for phoneme recognition.
- **Comprehensive Feedback**: Combines GOP scoring with LLM-generated teaching suggestions.

---

## **References**

1. **GOP Methodology**  
   - Cao, X., Fan, Z., Svendsen, T., & Salvi, G. (2024). A Framework for Phoneme-Level Pronunciation Assessment Using CTC. *Proc. Interspeech 2024* (pp. 302-306).

2. **LLM Feedback Generation**  
   - Zhong, H., Xie, Y., & Yao, Z. (2024). Leveraging Large Language Models to Refine Automatic Feedback Generation at Articulatory Level in Computer-Aided Pronunciation Training. *Proc. Interspeech 2024* (pp. 2600-2604).

---

## **Setup**

### Step 1: Install Dependencies
Follow these steps to set up the environment:
1.  conda environment
```
conda create -n wav2vec2-mdd python==3.8.0
conda activate wav2vec2-mdd
```

2. install requirements
```
pip install -r requirements.txt
```

3. torch cuda version ( see https://pytorch.org/get-started/previous-versions) )  
```
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
```


4. Assuming you've already installed HuggingFace transformers library, you need also to install the ctcdecode library
```
git clone --recursive https://github.com/parlance/ctcdecode.git
cd ctcdecode && pip install .
```

### Step 2: Prepare the Model
1. Download the pre-trained model:
   - [model.zip](https://140.122.184.167:5567/sharing/qRaWMnSBC)
2. Place the downloaded file into the directory:
   - `models/mdd/`
3. Navigate to the directory:
   ```bash
   cd models/mdd
   ```
4. Unzip the file:
   ```bash
   unzip wav2vec2-mdd.zip
   ```

---

## **Usage**

To start the MDD service, use the following command:
```bash
python app.py --timeout 12000
```

---

## **Testing**

To test the service, execute:
```bash
python client.py
```

---

## ** Acknowledgement **
This project was made possible through the contributions of:
- Fu-An Chao (Wav2vec2-mdd)
- Yu-Hsuan Fang (LLM Feedback)
