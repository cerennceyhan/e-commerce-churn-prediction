from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait 
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
import time
import random
import re
import os

from churn_prediction import paths

# ChromeDriver path: set CHROMEDRIVER_PATH, otherwise "chromedriver" from PATH is used.
DRIVER_PATH = os.environ.get('CHROMEDRIVER_PATH', 'chromedriver')
OUT_CSV = str(paths.SCRAPED_DATASET)

# Yorum linkini doğru bir şekilde oluşturan fonksiyon
def get_review_link(product_link):
    # Linki temel URL ve sorgu parametrelerine ayırır
    parts = product_link.split('?', 1)
    base_url = parts[0].rstrip('/')
    
    # Sorgu parametreleri varsa
    if len(parts) > 1:
        query_params = '?' + parts[1]
    else:
        query_params = ''
        
    # Yorumlar uzantısını sorgu parametrelerinden önce ekler
    return f"{base_url}/yorumlar{query_params}"

def load_product_links(path=paths.PRODUCT_LINKS):
    """Read product URLs (one per line, '#' comments allowed) from a text file."""
    with open(path, encoding='utf-8') as fh:
        return [ln.strip() for ln in fh if ln.strip() and not ln.lstrip().startswith('#')]


# Yorum yıldızı için padding değerleri
PADDING_TO_STAR = {66.8571: 1, 50.1429: 2, 33.4286: 3, 16.7143: 4, 0.0: 5}

def closest_star_from_padding(padding_value):
    if padding_value is None:
        return None
    closest_padding = min(PADDING_TO_STAR.keys(), key=lambda k: abs(k - padding_value))
    return PADDING_TO_STAR[closest_padding]

def extract_padding_value(style_str):
    if not style_str:
        return None
    m = re.search(r'padding-inline-end\s*:\s*([\d\.]+)px', style_str)
    if m:
        try:
            return float(m.group(1))
        except:
            return None
    return None

def main(links_file=paths.PRODUCT_LINKS):
    product_links = load_product_links(links_file)
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)

    # --- Chrome seçenekleri ---
    chrome_options = Options()
    chrome_options.add_argument("--start-maximized")
    chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36")

    # **********************************************
    # **** İNTERNET TASARRUFU İÇİN GEREKLİ AYARLAR ****
    # **********************************************
    prefs = {
        "profile.managed_default_content_settings.images": 2, # Resimleri engeller
        "profile.managed_default_content_settings.fonts": 2,  # Fontları engeller
    }
    chrome_options.add_experimental_option("prefs", prefs)
    # **********************************************

    service = Service(DRIVER_PATH)
    driver = webdriver.Chrome(service=service, options=chrome_options)
    # Açık bekleme nesnesi (Sayfanın yüklenmesini garantilemek için)
    wait = WebDriverWait(driver, 15) 

    try:
        for idx, link in enumerate(product_links, start=1):
            review_link = get_review_link(link)
        
            print(f"\n🔹 {idx}. ürün işleniyor. Hedef URL: {review_link}")
        
            # Sayfaya git
            driver.get(review_link)
        
            # ⚠️ KRİTİK ADIM: Her yeni link için sayfa yüklenmesini bekler.
            try:
                # Genel puan elemanının görünür olmasını bekleyerek sayfanın yüklendiğini kontrol et.
                wait.until(EC.visibility_of_element_located((By.CSS_SELECTOR, "div.summary-wrapper div.rate")))
            except:
                # Eğer puan yoksa, en azından bir yorum konteynırının yüklenmesini bekle.
                try:
                    wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "div.review-container")))
                except Exception as e:
                    print(f"⚠️ Uyarı: Kritik element (yorum konteynırı) yüklenemedi. Bu üründe yorum olmayabilir veya bot engeli var.")
                    # Bu ürünü atlayıp bir sonrakine geç
                    continue 
        
            # Ürünler arası bekleme (Bot tespitini azaltmak için artırıldı)
            time.sleep(4 + random.uniform(1, 3)) 

            # Trendyol'un otomatik olarak ana sayfaya yönlendirip yönlendirmediğini kontrol edin
            current_url = driver.current_url
            if current_url != review_link and not current_url.endswith("/yorumlar"):
                print(f"⚠️ Uyarı: Trendyol linki {current_url} olarak değiştirdi. Veri çekimi zor olabilir.")

            genel_puan = None
            # Marka ve Ürün açıklaması
            try:
                info_span = driver.find_element(By.CSS_SELECTOR, "span.info-title-text")
                marka = info_span.find_element(By.TAG_NAME, "b").text.strip()
                aciklama = info_span.text.replace(marka, "").strip()
            except:
                marka = ""
                aciklama = ""

            # Fiyat
            try:
                fiyat = driver.find_element(By.CSS_SELECTOR, "div.ty-plus-price-original-price").text.strip()
            except:
                try:
                    fiyat = driver.find_element(By.CSS_SELECTOR, "div.price-current-price[data-testid='current-price']").text.strip()
                except:
                    try:
                        fiyat = driver.find_element(By.CSS_SELECTOR, "span.prc-dsc").text.strip()
                    except:
                        fiyat = ""

            # Genel puan (5 üzerinden)
            try:
                rate_elem = driver.find_element(By.CSS_SELECTOR, "div.summary-wrapper div.rate")
                if rate_elem:
                    genel_puan = float(rate_elem.text.strip().replace(",", "."))
            except:
                pass

            # --- Yorumları yükle (ZORLAYICI SCROLL MANTIĞI) ---
            print("Yorumları yüklemek için zorlayarak kaydırılıyor...")
        
            SCROLL_PAUSE_TIME_MIN = 2 
            SCROLL_PAUSE_TIME_MAX = 4
            MAX_TOTAL_SCROLL_ATTEMPTS = 50 
        
            last_height = driver.execute_script("return document.body.scrollHeight")
            scroll_attempts = 0
        
            while scroll_attempts < MAX_TOTAL_SCROLL_ATTEMPTS: # Deneme sayısıyla sınırlı döngü
            
                # Sayfayı en aşağı kaydır
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            
                # Rastgele bekleme süresi (2-4 saniye)
                scroll_pause = random.uniform(SCROLL_PAUSE_TIME_MIN, SCROLL_PAUSE_TIME_MAX)
                time.sleep(scroll_pause) 
            
                new_height = driver.execute_script("return document.body.scrollHeight")
            
                if new_height == last_height:
                    # Yükseklik değişmediyse, bot algılamasını kırmak için ekstra bekleme.
                    print(f"-> Yükseklik stabil. Ekstra {2+random.uniform(0.5, 1.5):.2f}s bekleme...")
                    time.sleep(2 + random.uniform(0.5, 1.5)) 
                
                    final_height = driver.execute_script("return document.body.scrollHeight")
                
                    if final_height == new_height:
                        # İki kontrol sonrası da yükseklik değişmediyse, artık yorum gelmeyecektir.
                        print("-> Yükseklik hala stabil. Yorum yükleme sonlandırılıyor.")
                        break
                    
                last_height = new_height
                scroll_attempts += 1
            
            print(f"Kaydırma tamamlandı. Toplam deneme: {scroll_attempts}")


            # --- Yorum containerları ---
            review_containers = driver.find_elements(By.CSS_SELECTOR, "div.review")
            print(f"Toplam {len(review_containers)} yorum bulundu.")

            data = []
            for rev in review_containers:
                # Kullanıcı adı
                try:
                    ad = rev.find_element(By.CSS_SELECTOR, "div.review-info-detail div.name").text.strip()
                except:
                    ad = None

                # Tarih
                try:
                    tarih = rev.find_element(By.CSS_SELECTOR, "div.review-info-detail div.date").text.strip().replace("\n"," ")
                except:
                    tarih = None

                # Boy/Kilo/Beden
                boy = kilo = beden = None
                try:
                    variant_divs = rev.find_elements(By.CSS_SELECTOR, "div.product-variant div.product-attribute-product-attribute")
                    for v in variant_divs:
                        label = v.find_element(By.CSS_SELECTOR, "span.product-attribute-label").text.strip().lower()
                        value = v.find_element(By.CSS_SELECTOR, "span.product-attribute-value").text.strip()
                        if "boy" in label:
                            boy = value
                        elif "kilo" in label:
                            kilo = value
                        elif "beden" in label:
                            beden = value
                except:
                    pass

                # Yorum metni
                try:
                    comment = rev.find_element(By.CSS_SELECTOR, "div.review-comment span.review-comment").text.strip()
                except:
                    comment = ""

                # Satıcı
                try:
                    satıcı = rev.find_element(By.CSS_SELECTOR, "div.comment-seller-info span.seller-name-wrapper strong").text.strip()
                except:
                    satıcı = None

                # Kullanıcının verdiği puan (5 üzerinden)
                puan = None
                try:
                    star_elem = rev.find_element(By.CSS_SELECTOR, "div.star-rating-full-star")
                    style = star_elem.get_attribute("style")
                    padding_val = extract_padding_value(style)
                    puan = closest_star_from_padding(padding_val)
                except:
                    puan = None

                data.append({
                    "Marka": marka,
                    "Ürün": aciklama,
                    "Fiyat": fiyat,
                    "Genel Puan": genel_puan,
                    "Ad": ad,
                    "Yorum": comment,
                    "Tarih": tarih,
                    "Boy": boy,
                    "Kilo": kilo,
                    "Beden": beden,
                    "Satıcı": satıcı,
                    "Puan": puan
                })

            # --- CSV'ye kaydet ---
            df = pd.DataFrame(data)
            if os.path.exists(OUT_CSV):
                df.to_csv(OUT_CSV, mode='a', index=False, header=False, encoding="utf-8-sig")
            else:
                df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
            print(f"✅ {len(df)} yorum CSV'ye kaydedildi: {OUT_CSV}")

    finally:
        driver.quit()
        print("\n🚪 Tarayıcı kapatıldı.")


if __name__ == "__main__":
    main()
