from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, StaleElementReferenceException
import time
import random
from typing import List, Dict, Optional
import logging
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PartSelectScraper:
    """Complete scraper based on friend's working implementation"""
    
    BASE_URL = "https://www.partselect.com"
    
    def __init__(self):
        self.driver = self._setup_driver()
    
    def _setup_driver(self):
        """Setup Chrome driver - friend's configuration"""
        chrome_options = Options()
        
        user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36'
        chrome_options.add_argument(f'user-agent={user_agent}')
        
        chrome_options.add_argument('--accept-language=en-US,en;q=0.9')
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument("--disable-gpu")
        
        import tempfile
        import os
        temp_dir = os.path.join(tempfile.gettempdir(), f"chrome_temp_{os.getpid()}")
        chrome_options.add_argument(f"--user-data-dir={temp_dir}")
        
        # Anti-detection
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option('useAutomationExtension', False)
        chrome_options.page_load_strategy = 'normal'
        
        driver = webdriver.Chrome(options=chrome_options)
        
        driver.execute_cdp_cmd('Network.setUserAgentOverride', {"userAgent": user_agent})
        driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
            "source": "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
        })
        
        driver.set_page_load_timeout(60)
        driver.set_script_timeout(30)
        driver.implicitly_wait(20)
        
        return driver
    
    def _safe_navigate(self, url: str, max_retries=3) -> bool:
        """Friend's safe navigation with retries"""
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    delay = random.uniform(3, 7)
                    time.sleep(delay)
                
                self.driver.get(url)
                
                # Wait for page load
                wait = WebDriverWait(self.driver, 30)
                wait.until(lambda d: d.execute_script('return document.readyState') == 'complete')
                
                # Check for access denied
                if "Access Denied" in self.driver.title:
                    if attempt < max_retries - 1:
                        logger.warning("Access denied, retrying...")
                        continue
                    return False
                
                # Verify body element
                try:
                    wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))
                    return True
                except TimeoutException:
                    if attempt < max_retries - 1:
                        continue
                    return False
                
            except Exception as e:
                logger.error(f"Navigation error (attempt {attempt+1}): {e}")
                if attempt == max_retries - 1:
                    return False
        
        return False
    
    def _safe_get_text(self, element):
        """Safely get text from element"""
        try:
            return element.text.strip()
        except:
            return ""
    
    def _safe_get_attribute(self, element, attribute):
        """Safely get attribute from element"""
        try:
            return element.get_attribute(attribute)
        except:
            return ""
    
    def scrape_all_data(self, max_parts=200, max_repairs=100):
        """Main entry point"""
        logger.info("Starting PartSelect scraping...")
        
        try:
            parts = self.scrape_parts(max_parts)
            repairs = self.scrape_repairs(max_repairs)
            
            return {'parts': parts, 'repairs': repairs}
        finally:
            self.driver.quit()
    
    def scrape_parts(self, max_parts=200) -> List[Dict]:
        """Scrape parts using friend's exact approach"""
        all_parts = []
        
        appliances = [
            ('refrigerator', 'https://www.partselect.com/Refrigerator-Parts.htm'),
            ('dishwasher', 'https://www.partselect.com/Dishwasher-Parts.htm')
        ]
        
        for appliance_type, base_url in appliances:
            if len(all_parts) >= max_parts:
                break
            
            logger.info(f"\nScraping {appliance_type} parts from {base_url}")
            
            # Navigate to main page
            if not self._safe_navigate(base_url):
                logger.error(f"Failed to load {base_url}")
                continue
            
            # Get brand links (friend's approach)
            brand_links = self._get_brand_links()
            logger.info(f"Found {len(brand_links)} brand categories")
            
            # Process first 4 brands to save time
            for idx, brand_url in enumerate(brand_links[:4], 1):
                if len(all_parts) >= max_parts:
                    break
                
                logger.info(f"Processing brand {idx}/4: {brand_url}")
                
                if not self._safe_navigate(brand_url):
                    continue
                
                # Get part info from category page
                part_list = self._get_parts_from_category()
                logger.info(f"  Found {len(part_list)} parts in category")
                
                for part_idx, (part_name, part_url) in enumerate(part_list[:25], 1):
                    if len(all_parts) >= max_parts:
                        break
                    
                    logger.info(f"  Scraping part {part_idx}/{min(25, len(part_list))}: {part_name}")
                    
                    part_data = self._scrape_part_details(part_name, part_url, appliance_type)
                    
                    if part_data:
                        all_parts.append(part_data)
                    
                    # Return to category page
                    self._safe_navigate(brand_url)
                    time.sleep(random.uniform(1, 2))
                
                time.sleep(random.uniform(2, 4))
        
        logger.info(f"Total parts scraped: {len(all_parts)}")
        return all_parts[:max_parts]
    
    def _get_brand_links(self) -> List[str]:
        """Get brand category links - friend's code"""
        brand_links = []
        
        try:
            wait = WebDriverWait(self.driver, 10)
            wait.until(EC.presence_of_element_located((By.CLASS_NAME, "nf__links")))
            
            ul_tags = self.driver.find_elements(By.CLASS_NAME, "nf__links")
            
            if ul_tags:
                li_tags = ul_tags[0].find_elements(By.TAG_NAME, "li")
                
                for li in li_tags:
                    try:
                        a_tag = li.find_element(By.TAG_NAME, "a")
                        href = a_tag.get_attribute("href")
                        if href:
                            brand_links.append(href)
                    except:
                        continue
        except Exception as e:
            logger.error(f"Error getting brand links: {e}")
        
        return brand_links
    
    def _get_parts_from_category(self) -> List[tuple]:
        """Get part list from category page - friend's approach"""
        part_list = []
        
        try:
            wait = WebDriverWait(self.driver, 10)
            wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "div.nf__part.mb-3")))
            
            part_divs = self.driver.find_elements(By.CSS_SELECTOR, "div.nf__part.mb-3")
            
            for part_div in part_divs:
                try:
                    a_tag = part_div.find_element(By.CLASS_NAME, "nf__part__detail__title")
                    span = a_tag.find_element(By.TAG_NAME, "span")
                    part_name = self._safe_get_text(span)
                    href = self._safe_get_attribute(a_tag, "href")
                    
                    if part_name and href:
                        part_list.append((part_name, href))
                except:
                    continue
        
        except Exception as e:
            logger.error(f"Error getting parts from category: {e}")
        
        return part_list
    
    def _scrape_part_details(self, part_name: str, product_url: str, appliance_type: str) -> Optional[Dict]:
        """Scrape detailed part info - friend's exact implementation"""
        
        data = {
            'part_type': part_name,
            'part_number': 'N/A',
            'man_part_number': '',
            'price': 0.0,
            'symptoms': [],
            'appliance_type': appliance_type,
            'replaces_part_numbers': [],
            'brand': 'N/A',
            'stock_status': 'N/A',
            'installation_video_url': '',
            'item_url': product_url
        }
        
        # Navigate to product page
        if not self._safe_navigate(product_url):
            logger.warning(f"Failed to navigate to {product_url}")
            return None
        
        try:
            # Wait for product page
            wait = WebDriverWait(self.driver, 10)
            wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "div.pd__wrap")))
            
            part_match = re.search(r'(PS\d+)', product_url)
            if part_match:
                data['part_number'] = part_match.group(1)

            try:
                mpn_elem = self.driver.find_element(By.CSS_SELECTOR, "span[itemprop='mpn']")
                data['man_part_number'] = self._safe_get_text(mpn_elem)
            except:
                pass

            try:
                brand_elem = self.driver.find_element(By.CSS_SELECTOR, "span[itemprop='brand'] span[itemprop='name']")
                data['brand'] = self._safe_get_text(brand_elem)
            except:
                pass
            
            try:
                avail_elem = self.driver.find_element(By.CSS_SELECTOR, "span[itemprop='availability']")
                data['stock_status'] = self._safe_get_text(avail_elem)
            except:
                pass
            
            try:
                video_container = self.driver.find_element(By.CSS_SELECTOR, "div.yt-video")
                video_id = self._safe_get_attribute(video_container, "data-yt-init")
                if video_id:
                    data['installation_video_url'] = f"https://www.youtube.com/watch?v={video_id}"
            except:
                pass
            
            try:
                price_container = self.driver.find_element(By.CSS_SELECTOR, "span.price.pd__price")
                time.sleep(1)  # Wait for dynamic price
                
                # Try js-partPrice
                try:
                    price_elem = price_container.find_element(By.CSS_SELECTOR, "span.js-partPrice")
                    price_text = self._safe_get_text(price_elem).replace('$', '').replace(',', '')
                    if price_text:
                        data['price'] = float(price_text)
                except:
                    # Try content attribute
                    price_content = self._safe_get_attribute(price_container, "content")
                    if price_content:
                        data['price'] = float(price_content)
            except:
                pass
            
            try:
                replace_elem = self.driver.find_element(By.CSS_SELECTOR, "div[data-collapse-container='{\"targetClassToggle\":\"d-none\"}']")
                replace_text = self._safe_get_text(replace_elem)
                # Extract part numbers
                replace_parts = re.findall(r'PS\d+', replace_text)
                data['replaces_part_numbers'] = replace_parts
            except:
                pass
            
            try:
                pd_wrap = self.driver.find_element(By.CSS_SELECTOR, "div.pd__wrap.row")
                info_divs = pd_wrap.find_elements(By.CSS_SELECTOR, "div.col-md-6.mt-3")
                
                for div in info_divs:
                    try:
                        header = div.find_element(By.CSS_SELECTOR, "div.bold.mb-1")
                        header_text = self._safe_get_text(header)
                        
                        if "This part fixes the following symptoms:" in header_text:
                            symptoms_text = div.text.replace(header_text, "").strip()
                            # Split by newlines or commas
                            symptoms = [s.strip() for s in re.split(r'[,\n]', symptoms_text) if s.strip()]
                            data['symptoms'] = symptoms[:10]  # Limit to 10
                        
                        elif "This part works with the following products:" in header_text:
                            # Could extract product types here if needed
                            pass
                    except:
                        continue
            except:
                pass
            
            logger.debug(f"Scraped: {data['part_number']} - ${data['price']}")
            return data
            
        except Exception as e:
            logger.error(f"Error scraping part details from {product_url}: {e}")
            return None
    
    def scrape_repairs(self, max_repairs=100) -> List[Dict]:
        """Scrape repair guides - friend's exact approach"""
        all_repairs = []
        
        appliances = [
            ('refrigerator', 'https://www.partselect.com/Repair/Refrigerator/'),
            ('dishwasher', 'https://www.partselect.com/Repair/Dishwasher/')
        ]
        
        for appliance_type, repair_url in appliances:
            if len(all_repairs) >= max_repairs:
                break
            
            logger.info(f"\nScraping {appliance_type} repairs from {repair_url}")
            
            if not self._safe_navigate(repair_url):
                logger.error(f"Failed to load {repair_url}")
                continue
            
            try:
                wait = WebDriverWait(self.driver, 30)
                symptom_list = wait.until(
                    EC.presence_of_element_located((By.CLASS_NAME, "symptom-list"))
                )
                
                # Get all symptom links
                symptom_elements = symptom_list.find_elements(By.TAG_NAME, "a")
                logger.info(f"Found {len(symptom_elements)} symptoms")
                
                symptom_data_list = []
                for element in symptom_elements:
                    try:
                        title_elem = element.find_element(By.CLASS_NAME, "title-md")
                        symptom = self._safe_get_text(title_elem)
                        
                        # Description
                        desc_elem = element.find_element(By.TAG_NAME, "p")
                        description = self._safe_get_text(desc_elem)
                        
                        # URL
                        url = self._safe_get_attribute(element, "href")
                        
                        if symptom and url:
                            symptom_data_list.append({
                                'symptom': symptom,
                                'description': description,
                                'url': url
                            })
                    except:
                        continue
                
                for idx, symptom_data in enumerate(symptom_data_list[:50], 1):
                    if len(all_repairs) >= max_repairs:
                        break
                    
                    logger.info(f"Processing repair {idx}/{min(50, len(symptom_data_list))}: {symptom_data['symptom']}")
                    
                    full_url = symptom_data['url']
                    if not full_url.startswith('http'):
                        full_url = self.BASE_URL + full_url
                    
                    repair_details = self._get_repair_details(full_url)
                    
                    repair_entry = {
                        'appliance_type': appliance_type,
                        'symptom': symptom_data['symptom'],
                        'symptom_description': symptom_data['description'],
                        'parts': repair_details.get('parts', []),
                        'detailed_guide_url': full_url,
                        'video_tutorial_url': repair_details.get('video_url', '')
                    }
                    
                    all_repairs.append(repair_entry)
                    
                    time.sleep(random.uniform(2, 4))
                
            except Exception as e:
                logger.error(f"Error scraping repairs: {e}")
        
        logger.info(f"Total repairs scraped: {len(all_repairs)}")
        return all_repairs[:max_repairs]
    
    def _get_repair_details(self, url: str) -> Dict:
        """Get repair details from symptom page - friend's code"""
        details = {
            'parts': [],
            'video_url': ''
        }
        
        try:
            if not self._safe_navigate(url):
                return details
            
            # Wait for repair intro
            wait = WebDriverWait(self.driver, 10)
            wait.until(EC.presence_of_element_located((By.CLASS_NAME, "repair__intro")))
            
            try:
                part_links = self.driver.find_elements(By.CSS_SELECTOR, "div.repair__intro a.js-scrollTrigger")
                for link in part_links:
                    part_name = self._safe_get_text(link)
                    if part_name:
                        details['parts'].append(part_name)
            except:
                pass
            
            try:
                video_element = self.driver.find_element(By.CSS_SELECTOR, "div[data-yt-init]")
                video_id = self._safe_get_attribute(video_element, "data-yt-init")
                if video_id:
                    details['video_url'] = f"https://www.youtube.com/watch?v={video_id}"
            except:
                pass
            
        except Exception as e:
            logger.debug(f"Error getting repair details: {e}")
        
        return details

if __name__ == "__main__":
    scraper = PartSelectScraper()
    data = scraper.scrape_all_data(max_parts=100, max_repairs=50)
    
    print(f"\nResults:")
    print(f"Parts: {len(data['parts'])}")
    print(f"Repairs: {len(data['repairs'])}")
    
    if data['parts']:
        print(f"\nSample part:")
        print(data['parts'][0])
    
    if data['repairs']:
        print(f"\nSample repair:")
        print(data['repairs'][0])