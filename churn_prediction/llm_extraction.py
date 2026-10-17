"""
==================================================================================
LLM İLE ÖZELLİK MÜHENDİSLİĞİ (CLAUDE 4.5 SONNET)
==================================================================================
"""

import pandas as pd
import json
import anthropic
import time
from tqdm import tqdm
import os

from churn_prediction import paths

class LLMFeatureExtractor:
    """
    Claude 4.5 Sonnet kullanarak ürün yorumlarından özellik çıkarma
    Her ürün işlenince anında kaydeder!
    """
    
    def __init__(self, original_csv_path, product_features_csv_path, output_path, api_key):
        # Orijinal yorumları yükle
        self.df_reviews = pd.read_csv(original_csv_path, encoding='utf-8-sig')
        
        # Parse tarihleri
        self._parse_dates()
        
        # Ürün özelliklerini yükle (Phase 1 çıktısı)
        self.df_products = pd.read_csv(product_features_csv_path)
        
        # Claude client
        self.client = anthropic.Anthropic(api_key=api_key)
        
        # Output dosya yolu
        self.output_path = output_path
        
        # İşlenmiş ürünleri takip et
        self.processed_products = self._load_processed_products()
        
    def _parse_dates(self):
        """Tarihleri parse et"""
        month_mapping = {
            'Ocak': 'January', 'Şubat': 'February', 'Mart': 'March',
            'Nisan': 'April', 'Mayıs': 'May', 'Haziran': 'June',
            'Temmuz': 'July', 'Ağustos': 'August', 'Eylül': 'September',
            'Ekim': 'October', 'Kasım': 'November', 'Aralık': 'December'
        }
        
        def convert_date(date_str):
            if pd.isna(date_str):
                return None
            for tr, en in month_mapping.items():
                date_str = date_str.replace(tr, en)
            try:
                return pd.to_datetime(date_str, format='%d %B %Y')
            except:
                return None
        
        self.df_reviews['parsed_date'] = self.df_reviews['Tarih'].apply(convert_date)
        self.df_reviews = self.df_reviews.dropna(subset=['parsed_date'])
    
    def _load_processed_products(self):
        """
        Daha önce işlenmiş ürünleri yükle (kaldığı yerden devam için)
        """
        if os.path.exists(self.output_path):
            df_existing = pd.read_csv(self.output_path)
            processed = set(df_existing['Ürün'].unique())
            print(f"📂 Mevcut dosya bulundu: {len(processed)} ürün zaten işlenmiş")
            return processed
        return set()
    
    def _calculate_risk_class(self, row_dict):
        """
        Tek bir ürün için Risk_Class VE Risk_Score hesapla
        Returns: (risk_class, risk_score)
        """
        # Phase 1'den Toplam_Yorum_Sayisi al
        product_name = row_dict.get('Ürün')
        product_info = self.df_products[self.df_products['Ürün'] == product_name]
        
        if len(product_info) == 0:
            return 0, 0  # Default: Healthy, score 0
        
        toplam_yorum = product_info['Toplam_Yorum_Sayisi'].iloc[0]
        
        # 1. ENGAGEMENT CHURN (az yorum)
        if toplam_yorum < 5:
            return 2, 0  # Engagement churn, score 0 (yorum yetersiz)
        
        # 2. QUALITY CHURN (LLM özellikleri)
        quality_risk = 0
        
        # Kalıp problemi VAR ve ciddi
        if row_dict.get('fitment_problem') == True and row_dict.get('fitment_severity', 0) >= 7:
            quality_risk += 3
        elif row_dict.get('fitment_problem') == True:
            quality_risk += 1
        
        # Kumaş kalitesi problemi VAR
        if row_dict.get('fabric_quality_issue') == True:
            quality_risk += 2
        
        # LLM kalite algısı düşük
        quality_sentiment = row_dict.get('quality_sentiment', 5)
        if quality_sentiment <= 2:
            quality_risk += 3
        elif quality_sentiment == 3:
            quality_risk += 1
        
        # Teslimat problemi VAR
        if row_dict.get('delivery_issue') == True:
            quality_risk += 1
        
        # 3. SINIFLANDIRMA
        if quality_risk >= 4:
            return 1, quality_risk  # Quality Churn, score
        else:
            return 0, quality_risk  # Healthy, score
    
    def _save_single_result(self, result_dict):
        """
        TEK BİR ürünün sonucunu ANINDA kaydet!
        """
        # Risk_Class ve Risk_Score hesapla ve ekle
        risk_class, risk_score = self._calculate_risk_class(result_dict)
        result_dict['Risk_Class'] = risk_class
        result_dict['Risk_Score'] = risk_score
        
        # Yeni satırı DataFrame'e çevir
        new_row = pd.DataFrame([result_dict])
        
        # Dosya varsa append, yoksa create
        if os.path.exists(self.output_path):
            # Mevcut veriyi oku
            df_existing = pd.read_csv(self.output_path)
            # Yeni satırı ekle
            df_updated = pd.concat([df_existing, new_row], ignore_index=True)
            # Kaydet
            df_updated.to_csv(self.output_path, index=False, encoding='utf-8-sig')
        else:
            # İlk kez oluştur
            new_row.to_csv(self.output_path, index=False, encoding='utf-8-sig')
    
    def extract_product_comments(self, product_name, max_comments=100):
        """
        Bir ürüne ait yorumları çek
        Son yorumlara öncelik ver (daha güncel trendler)
        """
        product_reviews = self.df_reviews[
            self.df_reviews['Ürün'] == product_name
        ].sort_values('parsed_date', ascending=False).head(max_comments)
        
        # 'duzeltilmis_yorum' (spell-corrected text) only exists in the private dataset;
        # the public sample dataset just has the raw 'Yorum' column.
        column = 'duzeltilmis_yorum' if 'duzeltilmis_yorum' in product_reviews.columns else 'Yorum'
        return product_reviews[column].tolist()
    
    def create_llm_prompt(self, comments_list):
        """
        Claude için DÜZELTİLMİŞ prompt - Bilimsel eşiklerle
        """
        # Yorumları birleştir
        comments_text = "\n".join([f"- {comment}" for comment in comments_list if str(comment) != 'HATA' and pd.notna(comment)])
        
        prompt = f"""Bir e-ticaret ürününe ait kullanıcı yorumlarını analiz ediyorsun. Görevin, GENEL eğilimi belirlemek (birkaç aykırı yorumu değil).

YORUMLAR:
{comments_text}

Aşağıdaki bilgileri JSON formatında çıkar. SADECE JSON çıktısı ver, başka açıklama ekleme:

{{
  "fitment_problem": true/false,
  "fitment_severity": 0-10,
  "quality_sentiment": 1-5,
  "delivery_issue": true/false,
  "color_mismatch": true/false,
  "main_complaint": "string",
  "fabric_quality_issue": true/false,
  "price_value_perception": 1-5
}}

KRİTİK KURALLAR - ÇOK ÖNEMLİ:

1. fitment_problem: SADECE yorumların %20'sinden FAZLASI (5'te 1'i) beden/kalıp problemi belirtiyorsa TRUE. 
   Örnek: 100 yorumda 20'den fazlası "büyük/küçük/bol/dar" diyorsa TRUE, değilse FALSE.

2. fabric_quality_issue: SADECE yorumların %20'sinden FAZLASI kumaş kalitesinden şikayet ediyorsa TRUE.
   Örnek: 100 yorumda 20'den fazlası "kumaş kötü/ince/kalitesiz" diyorsa TRUE, değilse FALSE.

3. delivery_issue: SADECE yorumların %20'sinden FAZLASI teslimat sorunu belirtiyorsa TRUE.

4. color_mismatch: SADECE yorumların %20'sinden FAZLASI renk uyumsuzluğu belirtiyorsa TRUE.

5. quality_sentiment: ÇOĞUNLUĞUN genel kalite algısını yansıt.
   - Çoğunluk "mükemmel/harika/kaliteli" diyorsa → 5
   - Çoğunluk "iyi/güzel" diyorsa → 4
   - Çoğunluk "orta" diyorsa → 3
   - Çoğunluk "kötü" diyorsa → 2
   - Çoğunluk "berbat" diyorsa → 1

6. main_complaint: En sık tekrarlanan ciddi şikayeti yaz. Eğer ciddi şikayet yoksa "Genel memnuniyet yüksek" yaz.

7. fitment_severity & price_value_perception: 0-10 arası, GENEL eğilimi yansıt.

ÖRNEKLER:

Senaryo 1:
- 100 yorum
- 95 kişi: "Mükemmel, harika, bayıldım"
- 3 kişi: "Beden büyük geldi"
- 2 kişi: "Kumaş ince"
→ fitment_problem: FALSE (%3 < %20)
→ fabric_quality_issue: FALSE (%2 < %20)
→ quality_sentiment: 5
→ main_complaint: "Genel memnuniyet yüksek"

Senaryo 2:
- 100 yorum
- 30 kişi: "Beden çok büyük, kalıp kötü"
- 70 kişi: "Güzel ürün"
→ fitment_problem: TRUE (%30 > %20)
→ fitment_severity: 7 (ciddi problem)
→ quality_sentiment: 4 (çoğunluk memnun)
→ main_complaint: "Beden büyük geliyor"

SADECE YAYGIN SORUNLARI RAPORLA! Birkaç kişinin söylemesi SORUN DEĞİLDİR."""

        return prompt
    
    def call_llm_api(self, prompt, model="claude-sonnet-4-5-20250929"):
        """
        Claude API'ye istek gönder
        """
        try:
            response = self.client.messages.create(
                model=model,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Response'u temizle
            response_text = response.content[0].text.strip()
            
            # Markdown code block temizliği
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.startswith("```"):
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            
            response_text = response_text.strip()
            
            # JSON parse et
            result = json.loads(response_text)
            return result
        
        except json.JSONDecodeError as e:
            print(f"⚠️ JSON parse hatası: {e}")
            return None
        except Exception as e:
            print(f"⚠️ API hatası: {e}")
            return None
    
    def process_all_products(self, max_products=None, delay=1.0):
        """
        Tüm ürünler için LLM özelliklerini çıkar
        HER ÜRÜN İŞLENİNCE ANINDA KAYDEDER!
        """
        products = self.df_products['Ürün'].tolist()
        
        # Sadece işlenmemiş ürünleri al
        products_to_process = [p for p in products if p not in self.processed_products]
        
        if max_products:
            products_to_process = products_to_process[:max_products]
        
        print(f"\n🤖 LLM özellik çıkarma başlıyor...")
        print(f"   Toplam ürün: {len(products)}")
        print(f"   Zaten işlenmiş: {len(self.processed_products)}")
        print(f"   İşlenecek: {len(products_to_process)}")
      
        
        if len(products_to_process) == 0:
            print("\n✅ Tüm ürünler zaten işlenmiş!")
            return
        
        success_count = 0
        
        for product_name in tqdm(products_to_process, desc="Processing"):
            # Yorumları çek
            comments = self.extract_product_comments(product_name)
            
            if len(comments) == 0:
                print(f"⚠️ {product_name[:50]} için yorum bulunamadı")
                continue
            
            # Claude'a gönder
            prompt = self.create_llm_prompt(comments)
            llm_result = self.call_llm_api(prompt)
            
            if llm_result:
                # Ürün bilgilerini ekle
                llm_result['Ürün'] = product_name
                llm_result['Yorum_Sayisi'] = len(comments)
                
                # ANINDA KAYDET! 💾
                self._save_single_result(llm_result)
                
                # İşlenmiş olarak işaretle
                self.processed_products.add(product_name)
                success_count += 1
            
            # Rate limit
            time.sleep(delay)
        
        print(f"\n✅ {success_count} ürün için LLM özellikleri çıkarıldı ve kaydedildi")
    
    def merge_with_product_features(self):
        """
        LLM özelliklerini Phase 1'deki özelliklerle birleştir
        """
        if not os.path.exists(self.output_path):
            print("⚠️ Henüz hiç ürün işlenmemiş!")
            return None
        
        # LLM sonuçlarını oku
        df_llm = pd.read_csv(self.output_path)
        
        # Phase 1 ile birleştir
        df_final = self.df_products.merge(
            df_llm,
            on='Ürün',
            how='left'
        )
        
        return df_final
    
    def create_risk_class(self, df):
        """
        Risk_Class ve Risk_Score oluştur (SADECE LLM özellikleriyle)
        """
        def classify_product(row):
            # 1. ENGAGEMENT CHURN (az yorum)
            if row['Toplam_Yorum_Sayisi'] < 5:
                return 2, 0  # Engagement churn, score 0
            
            # 2. QUALITY CHURN (LLM özellikleri)
            quality_risk = 0
            
            # Kalıp problemi VAR ve ciddi
            if row['fitment_problem'] == True and row['fitment_severity'] >= 7:
                quality_risk += 3
            elif row['fitment_problem'] == True:
                quality_risk += 1
            
            # Kumaş kalitesi problemi VAR
            if row['fabric_quality_issue'] == True:
                quality_risk += 2
            
            # LLM kalite algısı düşük
            if row['quality_sentiment'] <= 2:
                quality_risk += 3
            elif row['quality_sentiment'] == 3:
                quality_risk += 1
            
            # Teslimat problemi VAR
            if row['delivery_issue'] == True:
                quality_risk += 1
            
            # 3. SINIFLANDIRMA
            if quality_risk >= 4:
                return 1, quality_risk  # Quality Churn
            else:
                return 0, quality_risk  # Healthy
        
        # Her satır için hem class hem score hesapla
        results = df.apply(classify_product, axis=1)
        df['Risk_Class'] = results.apply(lambda x: x[0])
        df['Risk_Score'] = results.apply(lambda x: x[1])
        
        # Dağılımı göster
        risk_dist = df['Risk_Class'].value_counts().sort_index()
        print(f"\n📊 HEDEF DEĞİŞKEN DAĞILIMI:")
        print(f"   Healthy (0):          {risk_dist.get(0, 0)} ürün ({risk_dist.get(0, 0)/len(df)*100:.1f}%)")
        print(f"   Quality Churn (1):    {risk_dist.get(1, 0)} ürün ({risk_dist.get(1, 0)/len(df)*100:.1f}%)")
        print(f"   Engagement Churn (2): {risk_dist.get(2, 0)} ürün ({risk_dist.get(2, 0)/len(df)*100:.1f}%)")
        
        print(f"\n📊 RISK_SCORE İSTATİSTİKLERİ:")
        print(f"   Ortalama: {df['Risk_Score'].mean():.2f}")
        print(f"   Min: {df['Risk_Score'].min()}")
        print(f"   Max: {df['Risk_Score'].max()}")
        
        return df
    
    def finalize_and_save(self, final_output_path):
        """
        Tüm işlem bittikten sonra final dosyayı oluştur
        """
        df_final = self.merge_with_product_features()
        
        if df_final is not None:
            # Risk_Class ekle
            df_final = self.create_risk_class(df_final)
            
            df_final.to_csv(final_output_path, index=False, encoding='utf-8-sig')
            
            print(f"\n💾 Final veri kaydedildi: {final_output_path}")
            print(f"\n📊 TOPLAM ÖZELLİK SAYISI: {len(df_final.columns)}")
            print("\n🔍 LLM ÖZELLİKLERİ:")
            llm_cols = ['fitment_problem', 'quality_sentiment', 'delivery_issue', 
                        'main_complaint', 'fabric_quality_issue', 'price_value_perception']
            for col in llm_cols:
                if col in df_final.columns:
                    print(f"   ✓ {col}")
            
            print(f"\n✓ Risk_Class eklendi!")
            
            return df_final
        
        return None


def get_api_key():
    """Read the Anthropic key from the environment (or a local .env file)."""
    try:
        from dotenv import load_dotenv
        load_dotenv(paths.PROJECT_ROOT / ".env")
    except ImportError:
        pass
    key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("CLAUDE_API_KEY")
    if not key:
        raise SystemExit(
            "ANTHROPIC_API_KEY is not set. Put it in a .env file (see .env.example) "
            "or export it in your shell."
        )
    return key


def main(max_products=None, delay=1.0):
    print("=" * 80)
    print("PHASE 2: LLM İLE ÖZELLİK MÜHENDİSLİĞİ")
    print("=" * 80)

    extractor = LLMFeatureExtractor(
        original_csv_path=str(paths.SAMPLE_DATASET),
        product_features_csv_path=str(paths.BASE_METRICS),
        output_path=str(paths.LLM_RESULTS),
        api_key=get_api_key(),
    )
    extractor.process_all_products(max_products=max_products, delay=delay)
    df_final = extractor.finalize_and_save(str(paths.LLM_EXTRACTION))

    print("\n" + "=" * 80)
    print("✅TAMAMLANDI!")
    return df_final


if __name__ == "__main__":
    main()
