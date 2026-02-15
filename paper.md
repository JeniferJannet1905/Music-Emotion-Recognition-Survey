# A Survey on Deep Learning Approaches for Music Emotion Recognition

Mr. S. Palanisamy
Assistant Professor
Department of Computer Applications
Bharathiar University, Coimbatore – 641046
Email: palsmailid@gmail.com

Jenifer Jannet J
Student, M.Sc. Data Analytics
Department of Computer Applications
Bharathiar University, Coimbatore, India
Email: jeniferjannet2004@gmail.com

# Abstract

Music has always been more than just sound; it is a universal language of emotion. This survey examines how the field of Music Emotion Recognition (MER), which aims to automatically detect and categorize emotional content in musical compositions, is being revolutionized by deep learning techniques. We review important datasets, feature extraction techniques, and model architectures such as Convolutional Neural Networks (CNNs), Recurrent Neural Networks (RNNs), Transformers, and hybrid systems in order to analyze the transition from conventional handcrafted features to complex neural architectures. Through our analysis, we highlight major advancements and persistent challenges, and suggest potential future research directions.

# Keywords

Music Emotion Recognition, Deep Learning, Convolutional Neural Network, BiLSTM, Transformer, Audio Feature Extraction, Multimodal Learning

# I. Introduction

Music has a profound impact on mood, behavior, and daily activities, and is deeply connected to human emotional experience. Emotional reactions ranging from joy and excitement to sadness and serenity are influenced by musical elements such as melody, rhythm, harmony, and tempo.

Music Emotion Recognition (MER) focuses on automatically identifying emotions conveyed through music. MER has applications in:
1) Intelligent human–computer interaction
2) Music recommendation systems
3) Mood-aware playlists
4) Music therapy
   
Early MER systems relied on traditional machine learning techniques combined with handcrafted features such as:
1) Mel-Frequency Cepstral Coefficients (MFCC)
2) Chroma features
3) Tempo
4) Spectral descriptors
However, these approaches struggled to capture complex musical structures and emotional dynamics.

With the introduction of deep learning, performance improved significantly:
1) CNNs capture frequency-based emotional patterns from spectrograms.
2) BiLSTM networks model temporal dependencies in music.
3) Transformers handle long-range dependencies and multimodal inputs.
4) Hybrid and multimodal systems combine spatial, temporal, and semantic learning.
Despite progress, challenges remain, including subjective emotion perception, limited labeled datasets, and high computational costs.

# II. Background and Fundamentals
A. Music Emotion Representation
MER typically follows two emotion modeling approaches:

Discrete Emotion Model
Classifies music into fixed categories:
1) Happy
2) Sad
3) Angry
4) Calm
5) Relaxed
Simple but limited in capturing gradual or mixed emotions.

Dimensional Emotion Model
The Valence–Arousal Model is widely used:
1) Valence → Positive or Negative emotion
2) Arousal → Energy or intensity level
Provides richer representation but increases annotation complexity.

B. Music Emotion Recognition Pipeline
A typical MER pipeline includes:
1) Input Acquisition – Audio, lyrics, metadata
2) Preprocessing – Noise removal, normalization, tokenization
3) Feature Extraction – Handcrafted or learned features
4) Model Learning – CNN, BiLSTM, Transformer, hybrid
5)Emotion Prediction – Categorical labels or valence–arousal scores

C. Evolution of MER Approaches
1) Early: Handcrafted audio features
2) Later: Context-aware approaches
3) Recent: Deep learning & multimodal modeling

D. Importance of Multimodal Information
Emotion is conveyed through:
1) Audio signals
2) Lyrics
Fusion strategies include:
1) Early Fusion
2) Late Fusion
3) Attention-Based Fusion

# III. Feature Extraction Techniques
A. Audio Feature Extraction
1. MFCC
Captures timbre-related spectral properties.
2. Chroma Features
Represent pitch class distribution and harmonic structure.
3. Spectral Contrast
Measures difference between spectral peaks and valleys.
4. Rhythm and Tempo Features
Fast tempo → High arousal
Slow tempo → Calm/Sad emotions

B. Text-Based Feature Extraction
1. Bag-of-Words
Simple but ignores context.
2. Word Embeddings
1) Word2Vec
2) GloVe
Capture semantic relationships.
3. Contextual Embeddings
Transformer-based models generate contextual representations.

C. Deep Feature Learning
1) CNN-Based Learning
Learns spatial features from spectrograms.
2) BiLSTM-Based Learning
Captures emotional progression over time.
3)Transformer-Based Learning
Uses attention mechanisms to model long-range dependencies.

D. Multimodal Feature Fusion
1) Early Fusion
Combine features before model input.
2) Late Fusion
Combine outputs of independent models.
3) Attention-Based Fusion
Dynamically assigns importance to modalities.

# IV. Datasets Used for Music Emotion Recognition

A. RAVDESS
Emotional speech and song dataset with discrete labels.
B. DEAM
Provides continuous valence–arousal annotations.
C. EMO-Music
Short instrumental excerpts with emotion labels.
D. Million Song Dataset (MSD)
Large-scale dataset for representation learning.
E. CAL500
Multi-label emotion dataset with tags and metadata.
F. PMEmo
Valence–arousal annotations with listener responses.

G. Dataset Challenges
1) Small dataset sizes
2) Subjective emotion labeling
3) Lack of benchmark standardization

# V. Deep Learning Models for MER
A. CNN Models
1) Input: Spectrograms
2) Strength: Spatial feature learning
3) Limitation: Poor long-term temporal modeling

B. RNN and BiLSTM Models
1) Capture sequential dependencies
2) Good for temporal emotion evolution
3) Slower training

C. Transformer Models
1) Attention-based
2) Handle long-range dependencies
3) High computational cost

D. Hybrid Models (CNN + BiLSTM)
1) Combine spatial + temporal learning
2) Higher accuracy
3) More complex

E. Multimodal Models
1) Audio + Lyrics
2) Better performance
3) Sensitive to noisy data

# VI. Comparative Analysis of Deep Learning Models
| Model Type  | Input Features           | Common Datasets    | Performance   | Advantages               | Limitations               |
| ----------- | ------------------------ | ------------------ | ------------- | ------------------------ | ------------------------- |
| CNN         | Spectrogram              | EMO-Music, CAL500  | Moderate–High | Fast, efficient          | Weak temporal modeling    |
| BiLSTM      | MFCC, Chroma             | RAVDESS, DEAM      | High          | Strong sequence modeling | Slow training             |
| Transformer | Audio + Lyrics           | DEAM, PMEmo        | High          | Long-range modeling      | High computation          |
| Hybrid      | Spectrogram + Sequential | EMO-Music, RAVDESS | Very High     | Combines strengths       | Complex architecture      |
| Multimodal  | Audio + Text             | CAL500, PMEmo      | Very High     | Rich emotional cues      | Sensitive to missing data |

# VII. Challenges and Limitations
1) Subjective emotion perception
2) Small datasets
3) High computational cost
4) Modality imbalance
5) Lack of standardized evaluation metrics

# VIII. Future Research Directions
1) Larger, diverse multilingual datasets
2) Improved annotation strategies
3) Self-supervised learning
4) Lightweight models for real-time applications
5) Better multimodal fusion techniques
6) Standardized benchmarking protocols

# IX. Conclusion

This survey reviewed recent developments in Music Emotion Recognition with a focus on deep learning approaches. CNNs, RNNs, Transformers, hybrid models, and multimodal systems significantly improve performance compared to traditional methods.

However, challenges such as subjective labeling, limited datasets, computational cost, and evaluation inconsistency remain.

Future research should emphasize diverse datasets, efficient architectures, and standardized evaluation frameworks to develop robust MER systems.

# References

1) X. Jiang et al., Music emotion recognition based on deep learning: A review, IEEE Access, 2024.
2) P. L. Louro et al., A comparison study of deep learning methodologies for music emotion recognition, Sensors, 2024.
3) X. Han et al., Music emotion recognition using neural networks, Electronics, 2023.
4) Z. Huang et al., Attention-based deep feature fusion, arXiv, 2022.
5) M. Malik et al., Stacked CNN-RNN for MER, arXiv, 2017.
6) Y. Qiao et al., Temporal convolutional attention network, Frontiers, 2024.
7) J. Qiu et al., Multi-task learning for MER, arXiv, 2022.
8) J. Dutta and D. Chanda, Comprehensive survey on MER, 2025.
9) S. Patil et al., Music emotion analysis review, 2025.
10) J. Kang and D. Herremans, Unified MER model, arXiv, 2025.
11) A. S. Sams, Multimodal MER in Indonesian songs, 2023.
12) R. Joy et al., Music mood recognition system, 2023.
13) A. Du, Applications of deep neural networks in MER, 2023.
14) S. Hizlisoy, CNN-LSTM for MER, 2021.
15) Y. Jia et al., Improved CNN for music emotion classification, 2022.
16) Real-time MER using multimodal fusion, 2025.
