import streamlit as st
from PIL import Image
import os
import logging
from model import FurnitureDetector

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configurar página
st.set_page_config(
    page_title="Furniture Finder",
    page_icon="🪑",
    layout="wide"
)

# Inicializar detector
@st.cache_resource
def get_detector():
    try:
        detector = FurnitureDetector()
        logger.info("Detector initialized successfully")
        return detector
    except Exception as e:
        logger.error(f"Error initializing detector: {str(e)}")
        return None

def analyze_image(image, detector, confidence_threshold):
    try:
        # Analisar imagem
        results = detector.detect_furniture(image, confidence_threshold)
        
        # Mostrar resultados
        st.image(results['image'], caption="Imagem com detecções")
        
        if results['detections']:
            st.success(f"Encontrados {len(results['detections'])} móveis!")
            
            # Mostrar detalhes de cada detecção
            for detection in results['detections']:
                with st.expander(f"🪑 {detection['class']} (Confiança: {detection['confidence']:.2f})"):
                    st.write(f"**Tipo:** {detection['class']}")
                    st.write(f"**Confiança:** {detection['confidence']:.2f}")
                    
                    # Buscar produtos similares
                    similar_products = detector.get_similar_products(detection['class'])
                    if similar_products:
                        st.write("**Produtos similares:**")
                        for store in similar_products:
                            st.markdown(f"- [{store['name']}]({store['url']})")
                    else:
                        st.info("Nenhum produto similar encontrado para esta categoria.")
        else:
            st.warning("Nenhum móvel detectado na imagem.")
            
    except Exception as e:
        logger.error(f"Error analyzing image: {str(e)}")
        st.error("Erro ao analisar a imagem. Por favor, tente novamente.")

def main():
    # Título e descrição
    st.title("🪑 Furniture Finder")
    st.markdown("""
    Faça upload de uma imagem de um móvel e descubra produtos similares em lojas portuguesas!
    """)
    
    # Sidebar
    with st.sidebar:
        st.header("Configurações")
        confidence_threshold = st.slider(
            "Limiar de confiança",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1
        )
        
        demo_mode = st.checkbox("Modo demo", value=True)
        if demo_mode:
            st.info("No modo demo, você verá resultados de exemplo.")
    
    # Inicializar detector
    detector = get_detector()
    if detector is None:
        st.error("Erro ao inicializar o detector. Por favor, tente novamente mais tarde.")
        return
    
    # Upload de imagem
    uploaded_file = st.file_uploader("Escolha uma imagem", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        try:
            # Carregar imagem
            image = Image.open(uploaded_file)
            
            if demo_mode:
                # Mostrar resultados de exemplo
                st.image(image, caption="Imagem carregada")
                st.success("Modo demo ativado - Mostrando resultados de exemplo")
                
                # Exemplo de detecções
                example_detections = [
                    {"class": "Chair", "confidence": 0.95},
                    {"class": "Table", "confidence": 0.88},
                    {"class": "Lamp", "confidence": 0.82},
                    {"class": "Sofa", "confidence": 0.78},
                    {"class": "Cabinet", "confidence": 0.75},
                    {"class": "Mirror", "confidence": 0.72},
                    {"class": "Shelf", "confidence": 0.68},
                    {"class": "Rug", "confidence": 0.65},
                    {"class": "Curtain", "confidence": 0.62},
                    {"class": "Pillow", "confidence": 0.60}
                ]
                
                for detection in example_detections:
                    with st.expander(f"🪑 {detection['class']} (Confiança: {detection['confidence']:.2f})"):
                        st.write(f"**Tipo:** {detection['class']}")
                        st.write(f"**Confiança:** {detection['confidence']:.2f}")
                        
                        # Exemplo de lojas
                        example_stores = [
                            {"name": "Zara Home", "url": "https://www.zarahome.com"},
                            {"name": "QuartoSala", "url": "https://www.quartosala.com"},
                            {"name": "Area Store", "url": "https://areastore.com"},
                            {"name": "IKEA Portugal", "url": "https://www.ikea.com/pt/pt/"}
                        ]
                        
                        st.write("**Produtos similares:**")
                        for store in example_stores:
                            st.markdown(f"- [{store['name']}]({store['url']})")
            else:
                # Analisar imagem real
                analyze_image(image, detector, confidence_threshold)
                
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            st.error("Erro ao processar a imagem. Por favor, tente novamente.")
    
    # Footer
    st.markdown("---")
    st.markdown("Desenvolvido com ❤️ por LEVisions")

if __name__ == "__main__":
    main() 