# Multilingual Text Classification

Tento projekt trénuje model pro **klasifikaci textů** ve více než **30 jazycích**. Dataset obsahuje články s popsanými kategoriemi (labely) a je veřejně dostupný na [Hugging Face](https://huggingface.co/datasets/marekk/testing_dataset_article_category).

## 🧠 Cíl projektu
Cílem je vytvořit klasifikátor, který dokáže správně určit kategorii textu bez ohledu na jazyk. Model využívá vícejazyčný transformer (např. **XLM-RoBERTa**).

## 📂 Dataset
- **Zdroj:** [marekk/testing_dataset_article_category](https://huggingface.co/datasets/marekk/testing_dataset_article_category)  
- **Počet jazyků:** až 30  
- **Formát:** CSV (sloupce `title`, `perex`, `label`)  
- **Popis:**
  - `title` – název článku  
  - `perex` – krátký úvod článku  
  - `label` – kategorie textu  
