"""The 30 dishes of 30VNFoods: folder name -> Vietnamese name and a short English description.

The descriptions are the prompts for CLIP's zero-shot classifier: an English-trained model
cannot be expected to know what "Bánh khọt" looks like, but it knows "mini crispy pancakes".
"""

DISHES: dict[str, tuple[str, str]] = {
    "Banh beo": ("Bánh bèo", "small steamed rice cakes in saucers topped with dried shrimp"),
    "Banh bot loc": ("Bánh bột lọc", "clear chewy tapioca dumplings with shrimp"),
    "Banh can": ("Bánh căn", "small round rice pancakes cooked in clay molds"),
    "Banh canh": ("Bánh canh", "thick noodle soup with udon-like noodles"),
    "Banh chung": ("Bánh chưng", "square sticky rice cake wrapped in green leaves"),
    "Banh cuon": ("Bánh cuốn", "steamed rice rolls with pork and fried shallots"),
    "Banh duc": ("Bánh đúc", "plain soft rice flour cake"),
    "Banh gio": ("Bánh giò", "pyramid-shaped rice dumpling wrapped in banana leaf"),
    "Banh khot": ("Bánh khọt", "mini crispy round pancakes topped with shrimp"),
    "Banh mi": ("Bánh mì", "Vietnamese baguette sandwich"),
    "Banh pia": ("Bánh pía", "flaky round pastry cake filled with mung bean and durian"),
    "Banh tet": ("Bánh tét", "cylindrical sticky rice cake wrapped in banana leaves"),
    "Banh trang nuong": ("Bánh tráng nướng", "grilled rice paper with egg, Vietnamese pizza"),
    "Banh xeo": ("Bánh xèo", "large crispy yellow crepe with shrimp and bean sprouts"),
    "Bun bo Hue": ("Bún bò Huế", "spicy red beef noodle soup from Hue"),
    "Bun dau mam tom": ("Bún đậu mắm tôm", "rice vermicelli with fried tofu and shrimp paste"),
    "Bun mam": ("Bún mắm", "dark fermented fish noodle soup"),
    "Bun rieu": ("Bún riêu", "crab and tomato noodle soup"),
    "Bun thit nuong": ("Bún thịt nướng", "rice vermicelli bowl with grilled pork and herbs"),
    "Ca kho to": ("Cá kho tộ", "caramelized fish braised in a clay pot"),
    "Canh chua": ("Canh chua", "sour soup with tamarind, pineapple and fish"),
    "Cao lau": ("Cao lầu", "Hoi An noodles with sliced pork and greens"),
    "Chao long": ("Cháo lòng", "rice porridge with pork offal"),
    "Com tam": ("Cơm tấm", "broken rice with grilled pork chop"),
    "Goi cuon": ("Gỏi cuốn", "fresh translucent spring rolls with shrimp and herbs"),
    "Hu tieu": ("Hủ tiếu", "clear pork broth noodle soup"),
    "Mi quang": ("Mì Quảng", "yellow turmeric noodles with a little broth"),
    "Nem chua": ("Nem chua", "fermented pork rolls wrapped in leaves"),
    "Pho": ("Phở", "Vietnamese beef noodle soup with rice noodles"),
    "Xoi xeo": ("Xôi xéo", "sticky rice with mung bean paste and fried shallots"),
}


def vietnamese_name(folder: str) -> str:
    return DISHES.get(folder, (folder, ""))[0]


def prompt(folder: str) -> str:
    name, desc = DISHES[folder]
    return f"a photo of {folder.lower()}, {desc}, a Vietnamese dish"
