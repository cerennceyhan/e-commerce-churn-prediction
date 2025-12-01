"""
==================================================================================
TEMEL ÖZELLİK HAZIRLAMA
==================================================================================
Bu script:
1. Her ürün için TEMEL metrikleri hesaplar
2. Sadece bağımsız özellikleri hazırlar
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
import os
warnings.filterwarnings('ignore')

class LeakFreeProductPreparator:

    
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path, encoding='utf-8-sig')
        self.product_features = None
        
    def parse_turkish_dates(self):
        """Türkçe tarihleri datetime'a çevir"""
        print("📅 Tarih parsing işlemi başlıyor...")
        
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
        
        self.df['parsed_date'] = self.df['Tarih'].apply(convert_date)
        self.df = self.df.dropna(subset=['parsed_date'])
        
        print(f"✅ {len(self.df):,} satır başarıyla tarih parse edildi")
        print(f"   Tarih Aralığı: {self.df['parsed_date'].min()} → {self.df['parsed_date'].max()}")
        
    def create_product_features(self):
        """
        Her ürün için TEMEL özellikleri oluştur
        ⚠️ Risk_Class burada OLUŞTURULMAZ!
        """
        print(f"\n🔧 Ürün özellikleri oluşturuluyor...")
        
        product_stats = []
        
        for product_name in self.df['Ürün'].unique():
            product_df = self.df[self.df['Ürün'] == product_name].copy()
            
            # ✅ SADECE BAĞIMSIZ ÖZELLİKLER
            stats = {
                'Ürün': product_name,
                'Marka': product_df['Marka'].iloc[0],
                
                # Genel metrikler
                'Genel_Puan': product_df['Genel Puan'].iloc[0],
                'Toplam_Yorum_Sayisi': len(product_df),
                'Puan_Standart_Sapma': product_df['Puan'].std(),
                'Min_Puan': product_df['Puan'].min(),
                'Max_Puan': product_df['Puan'].max(),
                
                # Puan dağılımı
                'Puan_5_Oran': (product_df['Puan'] == 5).sum() / len(product_df),
                'Puan_4_Oran': (product_df['Puan'] == 4).sum() / len(product_df),
                'Puan_3_Oran': (product_df['Puan'] == 3).sum() / len(product_df),
                'Puan_2_Oran': (product_df['Puan'] == 2).sum() / len(product_df),
                'Puan_1_Oran': (product_df['Puan'] == 1).sum() / len(product_df),
                
                # Negatif/Pozitif oranlar
                'Negatif_Yorum_Oran': (product_df['Puan'] <= 2).sum() / len(product_df),
                'Pozitif_Yorum_Oran': (product_df['Puan'] >= 4).sum() / len(product_df),
                
                # Yorum hızı (günlük)
                'Yorum_Hizi': self._calculate_review_velocity(product_df),
            }
            
            product_stats.append(stats)
        
        self.product_features = pd.DataFrame(product_stats)
        print(f"✅ {len(self.product_features)} ürün için özellikler oluşturuldu")
        
    def _calculate_review_velocity(self, product_df):
        """Günlük ortalama yorum sayısı"""
        date_range = (product_df['parsed_date'].max() - product_df['parsed_date'].min()).days
        if date_range == 0:
            return len(product_df)
        return len(product_df) / date_range
        
    def save_processed_data(self, output_path):
        """İşlenmiş veriyi kaydet"""
        self.product_features.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\n💾 Veri kaydedildi: {output_path}")
        
        # Özet istatistikler
        print(f"\n📈 ÖZET İSTATİSTİKLER:")
        print(self.product_features[['Genel_Puan', 'Negatif_Yorum_Oran', 
                                      'Puan_Standart_Sapma', 'Toplam_Yorum_Sayisi']].describe())
        
        return self.product_features


# ============================================================================
# KULLANIM ÖRNEĞİ
# ============================================================================
if __name__ == "__main__":
    
    # Proje kök dizinini bul
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Dosya yollarını göreceli olarak oluştur
    input_path = os.path.join(project_root, 'data', 'raw', 'sample_dataset.csv')
    output_path = os.path.join(project_root, 'data', 'processed', 'base_metrics.csv')
    
    preparator = LeakFreeProductPreparator(input_path)
    
    preparator.parse_turkish_dates()
    preparator.create_product_features()
    
    df_phase1 = preparator.save_processed_data(output_path)
    
 
    print("\n📌 SONRAKI ADIM:")
    print("python llm_extraction.py")