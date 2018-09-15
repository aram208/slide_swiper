# -*- coding: utf-8 -*-
from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


class SlideManager:
    def __init__(self, driverPath):
        self.driver = webdriver.Chrome(driverPath)
        self.action_chains = ActionChains(self.driver)

    def navigate_to(self, url, title):
        self.driver.get(url)
        if title is not None:
            assert title in self.driver.title

    def made_presentation_full_screen(self):
        wait = WebDriverWait(driver, 5)
        elemFullScreen = wait.until(EC.element_to_be_clickable((By.ID, 'punch-start-presentation-left')))
        elemFullScreen.click()

    def move_right(self):
        self.action_chains.send_keys(Keys.ARROW_RIGHT).perform()

    def move_left(self):
        self.action_chains.send_keys(Keys.ARROW_LEFT).perform()

    def close_webdriver(self):
        self.driver.close()