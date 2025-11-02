from dotenv import load_dotenv
import os
import pandas as pd
from azure.core.credentials import AzureKeyCredential
from azure.ai.textanalytics import TextAnalyticsClient

def main():
    try:
        # === 1. Cargar configuración desde .env ===
        load_dotenv()
        ai_endpoint = os.getenv('AI_SERVICE_ENDPOINT')
        ai_key = os.getenv('AI_SERVICE_KEY')

        if not ai_endpoint or not ai_key:
            raise ValueError(" No se encontró AI_SERVICE_ENDPOINT o AI_SERVICE_KEY en el archivo .env")

        # === 2. Crear cliente de Azure ===
        credential = AzureKeyCredential(ai_key)
        ai_client = TextAnalyticsClient(endpoint=ai_endpoint, credential=credential)

        # === 3. Definir carpeta de reseñas ===
        script_dir = os.path.dirname(os.path.abspath(__file__))
        reviews_folder = os.path.join(script_dir, 'reviewsADV')

        if not os.path.exists(reviews_folder):
            raise FileNotFoundError(f"❌ No se encontró la carpeta: {reviews_folder}")

        # Lista para guardar resultados
        resultados = []

        # === 4. Analizar cada reseña ===
        for file_name in os.listdir(reviews_folder):
            file_path = os.path.join(reviews_folder, file_name)
            with open(file_path, encoding='utf8') as f:
                text = f.read()

            print("\n-------------")
            print(file_name)
            print("\n" + text)

            # Detección de idioma
            detectedLanguage = ai_client.detect_language(documents=[text])[0]
            print("\nLanguage:", detectedLanguage.primary_language.name)

            # Análisis de sentimiento
            sentimentAnalysis = ai_client.analyze_sentiment(documents=[text])[0]
            print("\nSentiment:", sentimentAnalysis.sentiment)

            # Palabras clave
            key_phrases = ai_client.extract_key_phrases(documents=[text])[0].key_phrases
            if key_phrases:
                print("\nKey Phrases:")
                for phrase in key_phrases:
                    print("\t", phrase)

            # Entidades
            entities = ai_client.recognize_entities(documents=[text])[0].entities
            if entities:
                print("\nEntities")
                for entity in entities:
                    print(f"\t{entity.text} ({entity.category})")

            # Entidades con enlace
            linked_entities = ai_client.recognize_linked_entities(documents=[text])[0].entities
            if linked_entities:
                print("\nLinks")
                for linked_entity in linked_entities:
                    print(f"\t{linked_entity.name} ({linked_entity.url})")

            # Guardar resultados en lista
            resultados.append({
                "Archivo": file_name,
                "Idioma": detectedLanguage.primary_language.name,
                "Sentimiento": sentimentAnalysis.sentiment,
                "Key_Phrases": ", ".join(key_phrases),
                "Entities": ", ".join([f"{e.text} ({e.category})" for e in entities]),
                "Links": ", ".join([f"{le.name} ({le.url})" for le in linked_entities])
            })

        df_resultados = pd.DataFrame(resultados)
        output_path = os.path.join(script_dir, "resultados_text_analytics.xlsx")
        df_resultados.to_excel(output_path, index=False)

        print(f"\n Resultados guardados en: {output_path}")

    except Exception as ex:
        print("\n Error:", ex)

if __name__ == "__main__":
    main()
