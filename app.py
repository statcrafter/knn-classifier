"""
APPLICATION STREAMLIT - CLASSIFICATION D'IMAGES CIFAR-10
Déploie 2 modèles : KNN+HOG et CNN Optimisé
Auteur : ALLOUKOUTOU
"""

import streamlit as st
import numpy as np
from PIL import Image
import cv2
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
from skimage.feature import hog
import faiss

# Configuration de la page
st.set_page_config(
    page_title="CIFAR-10 Classifier",
    page_icon="🎯",
    layout="wide"
)

# ============================================================================
# CONSTANTES
# ============================================================================

CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

CLASS_NAMES_FR = ['avion', 'automobile', 'oiseau', 'chat', 'cerf', 
                  'chien', 'grenouille', 'cheval', 'bateau', 'camion']

# ============================================================================
# CHARGEMENT DES MODÈLES (avec cache)
# ============================================================================

@st.cache_resource
def load_knn_model():
    """Charge le modèle KNN+HOG"""
    try:
        with open('best_model_hog_faiss.pkl', 'rb') as f:
            model_data = pickle.load(f)
        
        # Reconstruire l'index FAISS
        dimension = model_data['X_train_all_pca'].shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(model_data['X_train_all_pca'])
        
        return {
            'scaler': model_data['scaler'],
            'pca': model_data['pca'],
            'index': index,
            'y_train': model_data['y_train_all'],
            'best_k': model_data['best_k'],
            'hog_params': model_data['hog_params'],
            'accuracy': model_data['test_accuracy'],
            'expected_features': model_data['expected_features']
        }
    except Exception as e:
        st.error(f"Erreur chargement KNN: {e}")
        return None

@st.cache_resource
def load_cnn_model():
    """Charge le modèle CNN"""
    try:
        model = load_model('best_cnn_model.h5')
        return model
    except Exception as e:
        st.error(f"Erreur chargement CNN: {e}")
        return None

# ============================================================================
# PRÉTRAITEMENT DES IMAGES
# ============================================================================

def preprocess_image(image, target_size=(32, 32)):
    """
    Prétraite une image pour les modèles CIFAR-10
    
    Args:
        image: PIL Image ou numpy array
        target_size: tuple (hauteur, largeur)
    
    Returns:
        numpy array de shape (32, 32, 3)
    """
    # Convertir PIL Image en numpy si nécessaire
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Convertir en RGB si nécessaire
    if len(image.shape) == 2:  # Grayscale
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:  # RGBA
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    
    # Resize à 32x32
    image_resized = cv2.resize(image, target_size, interpolation=cv2.INTER_LANCZOS4)
    
    return image_resized

def extract_hog_features(image, hog_params, expected_features):
    """
    Extrait les features HOG d'une image
    """
    # Convertir en grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Extraction HOG
    hog_features = hog(
        gray,
        orientations=hog_params['orientations'],
        pixels_per_cell=hog_params['pixels_per_cell'],
        cells_per_block=hog_params['cells_per_block'],
        block_norm=hog_params['block_norm'],
        visualize=False,
        feature_vector=True
    )
    
    # Features couleur
    color_features = []
    for c in range(3):
        channel = image[:, :, c].astype('float32') / 255.0
        color_features.extend([
            channel.mean(),
            channel.std(),
            np.percentile(channel, 25),
            np.percentile(channel, 75)
        ])
    
    # Combiner
    features = np.concatenate([hog_features, color_features])
    
    # Adapter aux dimensions attendues
    if len(features) > expected_features:
        features = features[:expected_features]
    elif len(features) < expected_features:
        features = np.pad(features, (0, expected_features - len(features)))
    
    return features

# ============================================================================
# PRÉDICTIONS
# ============================================================================

def predict_knn(image, model_data):
    """
    Prédiction avec KNN+HOG
    """
    # Prétraiter l'image
    img_processed = preprocess_image(image)
    
    # Extraire features HOG
    features = extract_hog_features(
        img_processed, 
        model_data['hog_params'],
        model_data['expected_features']
    )
    
    # Reshape
    features = features.reshape(1, -1)
    
    # Standardiser + PCA
    features_scaled = model_data['scaler'].transform(features)
    features_pca = model_data['pca'].transform(features_scaled).astype('float32')
    
    # Prédiction FAISS
    distances, indices = model_data['index'].search(features_pca, model_data['best_k'])
    
    # Vote majoritaire
    predictions = model_data['y_train'][indices[0]]
    unique, counts = np.unique(predictions, return_counts=True)
    
    # Calculer les probabilités (basées sur les votes)
    probabilities = np.zeros(10)
    for cls, count in zip(unique, counts):
        probabilities[cls] = count / model_data['best_k']
    
    prediction = unique[np.argmax(counts)]
    
    return prediction, probabilities

def predict_cnn(image, model):
    """
    Prédiction avec CNN
    """
    # Prétraiter l'image
    img_processed = preprocess_image(image)
    
    # Normaliser [0, 255] → [0, 1]
    img_normalized = img_processed.astype('float32') / 255.0
    
    # Ajouter dimension batch
    img_batch = np.expand_dims(img_normalized, axis=0)
    
    # Prédiction
    predictions = model.predict(img_batch, verbose=0)
    probabilities = predictions[0]
    prediction = np.argmax(probabilities)
    
    return prediction, probabilities

# ============================================================================
# INTERFACE STREAMLIT
# ============================================================================

def main():
    # Titre
    st.title("🎯 Classificateur d'Images CIFAR-10")
    st.markdown("### Projet Machine Learning - ISE")
    st.markdown("**Par : ALLOUKOUTOU Alex , KINKPE Judes & ALLOUKOUTOU Alex**")
    
    st.markdown("---")
    
    # Sidebar - Informations
    with st.sidebar:
        st.header("ℹ️ Informations")
        
        st.markdown("""
        ### Modèles disponibles
        
        **1. KNN + HOG** 🔵
        - K-Nearest Neighbors
        - Features HOG
        - Accuracy : 41%
        
        **2. CNN Optimisé** 🔴
        - Deep Learning
        - Architecture CNN
        - Accuracy : 91%
        
        ### Classes CIFAR-10
        """)
        
        for i, (en, fr) in enumerate(zip(CLASS_NAMES, CLASS_NAMES_FR)):
            st.markdown(f"{i}. {en} ({fr})")
        
        st.markdown("---")
        st.markdown("### 📚 À propos")
        st.markdown("""
        Projet de classification d'images utilisant :
        - Machine Learning classique (KNN)
        - Deep Learning (CNN)
        - Dataset CIFAR-10 (10 classes)
        """)
    
    # Chargement des modèles
    st.header("🔄 Chargement des modèles")
    
    col1, col2 = st.columns(2)
    
    with col1:
        with st.spinner("Chargement KNN+HOG..."):
            knn_model = load_knn_model()
            if knn_model:
                st.success(f"✅ KNN chargé (Accuracy: {knn_model['accuracy']*100:.2f}%)")
            else:
                st.error("❌ KNN non disponible")
    
    with col2:
        with st.spinner("Chargement CNN..."):
            cnn_model = load_cnn_model()
            if cnn_model:
                st.success("✅ CNN chargé (Accuracy: ~91%)")
            else:
                st.error("❌ CNN non disponible")
    
    st.markdown("---")
    
    # Upload d'image
    st.header("📤 Charger une image")
    
    uploaded_file = st.file_uploader(
        "Choisissez une image (JPG, PNG, JPEG)",
        type=['jpg', 'jpeg', 'png']
    )
    
    # OU utiliser une image d'exemple
    use_example = st.checkbox("Utiliser une image d'exemple CIFAR-10")
    
    if use_example:
        example_idx = st.slider("Choisir un exemple", 0, 9, 0)
        
        # Charger CIFAR-10 test
        from tensorflow.keras.datasets import cifar10
        (_, _), (X_test, y_test) = cifar10.load_data()
        
        # Prendre une image de la classe choisie
        class_indices = np.where(y_test == example_idx)[0]
        selected_idx = class_indices[0]
        
        image = X_test[selected_idx]
        true_label = y_test[selected_idx][0]
        
        st.info(f"Image d'exemple : **{CLASS_NAMES[true_label]}** ({CLASS_NAMES_FR[true_label]})")
    
    elif uploaded_file is not None:
        # Charger l'image uploadée
        image = Image.open(uploaded_file)
        image = np.array(image)
        true_label = None
    else:
        st.info("👆 Chargez une image ou utilisez un exemple")
        return
    
    # Afficher l'image
    st.markdown("---")
    st.header("🖼️ Image à classifier")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        # Afficher l'image originale
        st.image(image, caption="Image originale", use_container_width=True)
        
        # Afficher l'image prétraitée (32x32)
        img_preprocessed = preprocess_image(image)
        st.image(img_preprocessed, caption="Image prétraitée (32x32)", width=200)
    
    # Choix du modèle
    st.markdown("---")
    st.header("🤖 Sélectionner le modèle")
    
    model_choice = st.radio(
        "Choisissez le modèle de classification :",
        options=["KNN + HOG", "CNN Optimisé", "Les deux (comparaison)"],
        horizontal=True
    )
    
    # Bouton de prédiction
    if st.button("🚀 Classifier l'image", type="primary", use_container_width=True):
        
        st.markdown("---")
        st.header("📊 Résultats")
        
        if model_choice == "KNN + HOG" or model_choice == "Les deux (comparaison)":
            
            if knn_model:
                with st.spinner("Classification avec KNN+HOG..."):
                    knn_pred, knn_probs = predict_knn(image, knn_model)
                
                st.subheader("🔵 KNN + HOG")
                
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.metric(
                        label="Prédiction",
                        value=f"{CLASS_NAMES[knn_pred]}",
                        delta=f"{CLASS_NAMES_FR[knn_pred]}"
                    )
                    st.metric(
                        label="Confiance",
                        value=f"{knn_probs[knn_pred]*100:.1f}%"
                    )
                    
                    if true_label is not None:
                        if knn_pred == true_label:
                            st.success("✅ Correct !")
                        else:
                            st.error(f"❌ Incorrect (vrai: {CLASS_NAMES[true_label]})")
                
                with col2:
                    # Graphique des probabilités
                    import pandas as pd
                    import plotly.express as px
                    
                    df_knn = pd.DataFrame({
                        'Classe': [f"{CLASS_NAMES[i]}\n({CLASS_NAMES_FR[i]})" for i in range(10)],
                        'Probabilité': knn_probs * 100
                    })
                    
                    fig_knn = px.bar(
                        df_knn, 
                        x='Classe', 
                        y='Probabilité',
                        title='Distribution des probabilités (KNN)',
                        color='Probabilité',
                        color_continuous_scale='Blues'
                    )
                    fig_knn.update_layout(height=400)
                    st.plotly_chart(fig_knn, use_container_width=True)
        
        if model_choice == "CNN Optimisé" or model_choice == "Les deux (comparaison)":
            
            if model_choice == "Les deux (comparaison)":
                st.markdown("---")
            
            if cnn_model:
                with st.spinner("Classification avec CNN..."):
                    cnn_pred, cnn_probs = predict_cnn(image, cnn_model)
                
                st.subheader("🔴 CNN Optimisé")
                
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.metric(
                        label="Prédiction",
                        value=f"{CLASS_NAMES[cnn_pred]}",
                        delta=f"{CLASS_NAMES_FR[cnn_pred]}"
                    )
                    st.metric(
                        label="Confiance",
                        value=f"{cnn_probs[cnn_pred]*100:.1f}%"
                    )
                    
                    if true_label is not None:
                        if cnn_pred == true_label:
                            st.success("✅ Correct !")
                        else:
                            st.error(f"❌ Incorrect (vrai: {CLASS_NAMES[true_label]})")
                
                with col2:
                    # Graphique des probabilités
                    import pandas as pd
                    import plotly.express as px
                    
                    df_cnn = pd.DataFrame({
                        'Classe': [f"{CLASS_NAMES[i]}\n({CLASS_NAMES_FR[i]})" for i in range(10)],
                        'Probabilité': cnn_probs * 100
                    })
                    
                    fig_cnn = px.bar(
                        df_cnn, 
                        x='Classe', 
                        y='Probabilité',
                        title='Distribution des probabilités (CNN)',
                        color='Probabilité',
                        color_continuous_scale='Reds'
                    )
                    fig_cnn.update_layout(height=400)
                    st.plotly_chart(fig_cnn, use_container_width=True)
        
        # Comparaison si les deux modèles
        if model_choice == "Les deux (comparaison)" and knn_model and cnn_model:
            st.markdown("---")
            st.subheader("⚖️ Comparaison des modèles")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("KNN Prédiction", CLASS_NAMES[knn_pred])
            
            with col2:
                st.metric("CNN Prédiction", CLASS_NAMES[cnn_pred])
            
            with col3:
                if knn_pred == cnn_pred:
                    st.success("✅ Accord")
                else:
                    st.warning("⚠️ Désaccord")
            
            # Tableau comparatif
            import pandas as pd
            
            comparison_df = pd.DataFrame({
                'Modèle': ['KNN+HOG', 'CNN Optimisé'],
                'Prédiction': [CLASS_NAMES[knn_pred], CLASS_NAMES[cnn_pred]],
                'Confiance': [f"{knn_probs[knn_pred]*100:.1f}%", f"{cnn_probs[cnn_pred]*100:.1f}%"],
                'Accuracy': ['41%', '91%']
            })
            
            st.dataframe(comparison_df, use_container_width=True, hide_index=True)

if __name__ == "__main__":
    main()