import os
import re
import io
import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
import pandas as pd
import requests
from PIL import Image, ImageTk

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression

# ---------- Files ----------
PRODUCTS_CSV = "products.csv"
SOURCES_CSV = "sources.csv"
WISHLIST_CSV = "wishlist.csv"
FEEDBACK_CSV = "feedback.csv"
TOKEN_FILE = "token.txt"

# ---------- VK API ----------
VK_API_V = "5.199"


def load_or_empty(path, columns):
    if os.path.exists(path):
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.DataFrame(columns=columns)
    return pd.DataFrame(columns=columns)


def read_token():
    if os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE, "r", encoding="utf-8") as f:
            return f.read().strip()
    return ""


class WishRecommenderApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("VK Wish List + AI Recommender (Desktop)")
        self.geometry("1280x860")

        # Cache for Tk images to prevent garbage collection
        self._img_cache = {}

        # Ensure products.csv exists with needed columns
        if not os.path.exists(PRODUCTS_CSV):
            pd.DataFrame(columns=["id", "owner_id", "title", "description", "price", "url", "photo_url"]).to_csv(
                PRODUCTS_CSV, index=False, encoding="utf-8-sig"
            )

        self.token = read_token()

        self.products = pd.read_csv(PRODUCTS_CSV)
        for col in ["id", "owner_id", "title", "description", "price", "url", "photo_url"]:
            if col not in self.products.columns:
                self.products[col] = None

        self._normalize_products_df()

        self.sources = load_or_empty(SOURCES_CSV, ["owner_id"])
        if not self.sources.empty and "owner_id" in self.sources.columns:
            self.sources["owner_id"] = pd.to_numeric(self.sources["owner_id"], errors="coerce").fillna(0).astype(int)
        else:
            self.sources = pd.DataFrame(columns=["owner_id"])

        self.wishlist = load_or_empty(WISHLIST_CSV, ["owner_id", "id"])
        if not self.wishlist.empty:
            self.wishlist["owner_id"] = pd.to_numeric(self.wishlist["owner_id"], errors="coerce").fillna(0).astype(int)
            self.wishlist["id"] = pd.to_numeric(self.wishlist["id"], errors="coerce").fillna(0).astype(int)

        self.feedback = load_or_empty(FEEDBACK_CSV, ["owner_id", "id", "label"])
        if not self.feedback.empty:
            self.feedback["owner_id"] = pd.to_numeric(self.feedback["owner_id"], errors="coerce").fillna(0).astype(int)
            self.feedback["id"] = pd.to_numeric(self.feedback["id"], errors="coerce").fillna(0).astype(int)
            self.feedback["label"] = pd.to_numeric(self.feedback["label"], errors="coerce").fillna(0).astype(int)

        self.vectorizer = None
        self.X = None
        self._rebuild_tfidf()

        self._build_ui()
        self.refresh_sources_list()
        self.refresh_wishlist_tiles()
        self.refresh_recommendations()

    # ---------------- Utility ----------------
    def save_df(self, df, path):
        df.to_csv(path, index=False, encoding="utf-8-sig")

    def _normalize_products_df(self):
        self.products["title"] = self.products["title"].fillna("")
        self.products["description"] = self.products["description"].fillna("")
        self.products["photo_url"] = self.products["photo_url"].fillna("")
        self.products["url"] = self.products["url"].fillna("")

        if len(self.products) > 0:
            self.products["owner_id"] = pd.to_numeric(self.products["owner_id"], errors="coerce").fillna(0).astype(int)
            self.products["id"] = pd.to_numeric(self.products["id"], errors="coerce").fillna(0).astype(int)

    def _rebuild_tfidf(self):
        self._normalize_products_df()
        self.products["text"] = (self.products["title"] + " " + self.products["description"]).str.strip()

        if len(self.products) == 0:
            self.vectorizer = None
            self.X = None
            return

        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.X = self.vectorizer.fit_transform(self.products["text"].tolist())

    def _short(self, s: str, n: int = 90) -> str:
        s = (s or "").strip()
        if len(s) <= n:
            return s
        return s[: n - 1].rstrip() + "…"

    # ---------------- Clipboard + Context Menu ----------------
    def paste_from_clipboard(self):
        try:
            text = self.clipboard_get()
        except tk.TclError:
            text = ""
        if text:
            self.link_entry.delete(0, tk.END)
            self.link_entry.insert(0, text)

    def _show_entry_menu(self, event):
        self.entry_menu.tk_popup(event.x_root, event.y_root)

    # ---------------- VK API ----------------
    def vk_call(self, method: str, params: dict):
        if not self.token:
            raise RuntimeError(
                "Нет токена. Создай token.txt рядом с приложением и вставь туда VK access_token с правами market,groups."
            )
        url = f"https://api.vk.com/method/{method}"
        p = {"access_token": self.token, "v": VK_API_V}
        p.update(params)
        r = requests.get(url, params=p, timeout=25)
        data = r.json()
        if "error" in data:
            raise RuntimeError(str(data["error"]))
        return data["response"]

    def _best_photo_url(self, it: dict) -> str:
        if isinstance(it.get("thumb_photo"), str) and it.get("thumb_photo"):
            return it["thumb_photo"]

        photos = it.get("photos")
        if isinstance(photos, list) and photos:
            p0 = photos[0]
            sizes = p0.get("sizes") if isinstance(p0, dict) else None
            if isinstance(sizes, list) and sizes:
                best = ""
                best_area = -1
                for s in sizes:
                    if not isinstance(s, dict):
                        continue
                    w = s.get("width") or 0
                    h = s.get("height") or 0
                    u = s.get("url") or ""
                    area = int(w) * int(h)
                    if u and area > best_area:
                        best_area = area
                        best = u
                if best:
                    return best
        return ""

    def _normalize_vk_item(self, it: dict):
        price = None
        if isinstance(it.get("price"), dict):
            amount = it["price"].get("amount")
            if amount is not None:
                try:
                    price = int(amount) / 100
                except Exception:
                    price = None

        owner_id = int(it.get("owner_id"))
        item_id = int(it.get("id"))
        photo_url = self._best_photo_url(it)

        return {
            "id": item_id,
            "owner_id": owner_id,
            "title": it.get("title", "") or "",
            "description": it.get("description", "") or "",
            "price": price,
            "url": f"https://vk.com/market{owner_id}_{item_id}",
            "photo_url": photo_url,
        }

    def parse_vk_market_link(self, link: str):
        """
        Supports:
        1) https://vk.com/market-127356458_13579885
        2) https://vk.com/market/product/slug-69239501-12511354 (owner_id -> negative)
        """
        m1 = re.search(r"market(-?\d+)_(\d+)", link)
        m2 = re.search(r"market/product/.+-(\d+)-(\d+)", link)
        if m1:
            return int(m1.group(1)), int(m1.group(2))
        if m2:
            return -int(m2.group(1)), int(m2.group(2))
        return None, None

    def ensure_source(self, owner_id: int):
        if "owner_id" not in self.sources.columns:
            self.sources = pd.DataFrame(columns=["owner_id"])
        if owner_id not in self.sources["owner_id"].astype(int).tolist():
            self.sources = pd.concat([self.sources, pd.DataFrame([{"owner_id": owner_id}])], ignore_index=True)
            self.save_df(self.sources, SOURCES_CSV)

    def download_group_items(self, owner_id: int, limit: int = 200):
        """
        ВАЖНО: upsert — новые строки должны ПЕРЕЗАПИСЫВАТЬ старые,
        чтобы старым товарам подтянулись photo_url/описание/цена.
        """
        offset = 0
        step = 50
        new_rows = []

        while offset < limit:
            resp = self.vk_call(
                "market.get",
                {"owner_id": owner_id, "count": min(step, limit - offset), "offset": offset},
            )
            items = resp.get("items", [])
            if not items:
                break

            for it in items:
                row = self._normalize_vk_item(it)
                if str(row.get("title", "")).strip():
                    new_rows.append(row)

            offset += len(items)

        if not new_rows:
            return

        # --- UPSERT ---
        add_df = pd.DataFrame(new_rows)
        self.products = pd.concat([self.products, add_df], ignore_index=True)

        self._normalize_products_df()

        # keep='last' -> новые записи выигрывают
        self.products = self.products.drop_duplicates(subset=["owner_id", "id"], keep="last").copy()
        self.save_df(self.products, PRODUCTS_CSV)

        self._rebuild_tfidf()
        self._refresh_combo_values()

    # ---------------- Images ----------------
    def load_image_thumb(self, url: str, size=(160, 160)):
        if not url:
            return None
        key = (url, size)
        if key in self._img_cache:
            return self._img_cache[key]

        try:
            r = requests.get(url, timeout=15)
            r.raise_for_status()
            img = Image.open(io.BytesIO(r.content)).convert("RGB")
            img.thumbnail(size)
            tk_img = ImageTk.PhotoImage(img)
            self._img_cache[key] = tk_img
            return tk_img
        except Exception:
            return None

    # ---------------- Feedback helpers ----------------
    def get_feedback_label(self, owner_id: int, item_id: int):
        if self.feedback.empty:
            return None
        m = self.feedback[(self.feedback["owner_id"] == owner_id) & (self.feedback["id"] == item_id)]
        if m.empty:
            return None
        return int(m.iloc[0]["label"])

    def set_feedback(self, owner_id: int, item_id: int, label: int):
        mask = (self.feedback["owner_id"] == owner_id) & (self.feedback["id"] == item_id)
        if mask.any():
            self.feedback.loc[mask, "label"] = label
        else:
            self.feedback = pd.concat(
                [self.feedback, pd.DataFrame([{"owner_id": owner_id, "id": item_id, "label": label}])],
                ignore_index=True
            )
        self.save_df(self.feedback, FEEDBACK_CSV)

    # ---------------- UI ----------------
    def _build_ui(self):
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True)

        self.tab_wl = ttk.Frame(self.notebook)
        self.tab_rec = ttk.Frame(self.notebook)

        self.notebook.add(self.tab_wl, text="Wish List")
        self.notebook.add(self.tab_rec, text="Recommendations")

        self._build_wishlist_tab()
        self._build_recommend_tab()

    # ---------- Tab 1: Wish List (TILES) ----------
    def _build_wishlist_tab(self):
        top = ttk.Frame(self.tab_wl)
        top.pack(fill="x", padx=10, pady=10)

        link_frame = ttk.Frame(top)
        link_frame.pack(fill="x", pady=5)
        ttk.Label(link_frame, text="Вставь ссылку на товар VK Market:").pack(anchor="w")

        self.link_var = tk.StringVar()
        self.link_entry = ttk.Entry(link_frame, textvariable=self.link_var, width=120)
        self.link_entry.pack(side="left", fill="x", expand=True, pady=3)

        self.link_entry.bind("<Control-v>", lambda e: self.paste_from_clipboard() or "break")
        self.link_entry.bind("<Control-V>", lambda e: self.paste_from_clipboard() or "break")
        self.link_entry.bind("<Shift-Insert>", lambda e: self.paste_from_clipboard() or "break")

        self.entry_menu = tk.Menu(self, tearoff=0)
        self.entry_menu.add_command(label="Вставить", command=self.paste_from_clipboard)
        self.entry_menu.add_command(label="Очистить", command=lambda: self.link_entry.delete(0, tk.END))
        self.link_entry.bind("<Button-3>", self._show_entry_menu)

        ttk.Button(link_frame, text="Вставить", command=self.paste_from_clipboard).pack(side="left", padx=8)
        ttk.Button(link_frame, text="Добавить по ссылке", command=self.add_by_link).pack(side="left", padx=8)

        ttk.Separator(top).pack(fill="x", pady=8)

        ttk.Label(top, text="Или выбери товар из уже загруженного каталога:").pack(anchor="w")
        self.product_var = tk.StringVar()
        self.product_combo = ttk.Combobox(top, textvariable=self.product_var, state="readonly", width=120)
        self._refresh_combo_values()
        self.product_combo.pack(fill="x", pady=5)
        ttk.Button(top, text="Добавить выбранное", command=self.add_selected_to_wishlist).pack(anchor="w", pady=5)

        # Sources + refresh old goods photos
        src_frame = ttk.Frame(self.tab_wl)
        src_frame.pack(fill="x", padx=10, pady=10)

        ttk.Label(src_frame, text="Источники (сообщества):").pack(anchor="w")
        self.src_list = tk.Listbox(src_frame, height=5)
        self.src_list.pack(fill="x", expand=False, pady=4)

        btns = ttk.Frame(src_frame)
        btns.pack(fill="x", pady=3)
        ttk.Button(btns, text="Докачать товары выбранного источника (200)", command=self.download_selected_source).pack(side="left")
        ttk.Button(btns, text="Обновить ВСЕ источники (с фото)", command=self.refresh_all_sources).pack(side="left", padx=8)

        # WishList tiles area
        wl_title = ttk.Frame(self.tab_wl)
        wl_title.pack(fill="x", padx=10, pady=(5, 0))
        ttk.Label(wl_title, text="Твои хотелки (плитки):").pack(side="left")
        ttk.Button(wl_title, text="Обновить Wish List", command=self.refresh_wishlist_tiles).pack(side="right")

        self.wl_canvas = tk.Canvas(self.tab_wl, highlightthickness=0)
        self.wl_scroll = ttk.Scrollbar(self.tab_wl, orient="vertical", command=self.wl_canvas.yview)
        self.wl_canvas.configure(yscrollcommand=self.wl_scroll.set)
        self.wl_scroll.pack(side="right", fill="y")
        self.wl_canvas.pack(side="left", fill="both", expand=True, padx=10, pady=10)

        self.wl_inner = ttk.Frame(self.wl_canvas)
        self.wl_window = self.wl_canvas.create_window((0, 0), window=self.wl_inner, anchor="nw")
        self.wl_inner.bind("<Configure>", lambda e: self.wl_canvas.configure(scrollregion=self.wl_canvas.bbox("all")))
        self.wl_canvas.bind("<Configure>", lambda e: self.wl_canvas.itemconfig(self.wl_window, width=e.width))

    def _refresh_combo_values(self):
        vals = []
        for row in self.products.itertuples():
            try:
                title = (row.title or "").strip()
                if not title:
                    title = f"(без названия) market{int(row.owner_id)}_{int(row.id)}"
                vals.append(f"{title} | market{int(row.owner_id)}_{int(row.id)}")
            except Exception:
                pass
        self.product_combo["values"] = vals
        if vals:
            self.product_combo.current(0)

    def refresh_sources_list(self):
        if not hasattr(self, "src_list"):
            return
        self.src_list.delete(0, tk.END)
        if self.sources.empty:
            self.src_list.insert(tk.END, "Источники пусты.")
            return
        for oid in self.sources["owner_id"].astype(int).tolist():
            self.src_list.insert(tk.END, str(oid))

    def download_selected_source(self):
        sel = self.src_list.curselection()
        if not sel:
            return
        txt = self.src_list.get(sel[0]).strip()
        if not re.match(r"^-?\d+$", txt):
            return
        owner_id = int(txt)
        try:
            self.download_group_items(owner_id, limit=200)
            messagebox.showinfo("Готово", f"Обновили товары из {owner_id} (и фото тоже, если VK API их отдаёт).")
        except Exception as e:
            messagebox.showerror("Ошибка VK API", str(e))

        self.refresh_wishlist_tiles()
        self.refresh_recommendations()

    def refresh_all_sources(self):
        if self.sources.empty:
            messagebox.showinfo("Источники пусты", "Сначала добавь хотя бы один источник (сообщество).")
            return
        try:
            # обновляем по 200 товаров с каждого источника
            for oid in self.sources["owner_id"].astype(int).tolist():
                self.download_group_items(int(oid), limit=200)
            messagebox.showinfo("Готово", "Обновили все источники. Фото у старых товаров должны появиться.")
        except Exception as e:
            messagebox.showerror("Ошибка VK API", str(e))

        self.refresh_wishlist_tiles()
        self.refresh_recommendations()

    def add_by_link(self):
        link = self.link_var.get().strip()
        if not link:
            messagebox.showwarning("Пустая ссылка", "Вставь ссылку на товар VK.")
            return

        owner_id, item_id = self.parse_vk_market_link(link)
        if owner_id is None:
            messagebox.showerror(
                "Ошибка",
                "Не удалось распознать ссылку VK Market.\n\n"
                "Рекомендуемый формат:\nhttps://vk.com/market-OWNERID_ITEMID\n"
                "(market/product/... может быть витринным и API его не отдаёт)"
            )
            return

        # add source and download items (upsert will fill photo_url for old goods too)
        self.ensure_source(owner_id)
        self.refresh_sources_list()

        # If item not in catalog -> pull this group
        exists = self.products[(self.products["owner_id"] == owner_id) & (self.products["id"] == item_id)]
        if exists.empty:
            try:
                self.download_group_items(owner_id, limit=200)
            except Exception as e:
                messagebox.showerror("Ошибка VK API", f"Не удалось скачать товары сообщества.\n\n{e}")
                return

        # Add to wishlist
        already = ((self.wishlist["owner_id"] == owner_id) & (self.wishlist["id"] == item_id)).any()
        if not already:
            self.wishlist = pd.concat(
                [self.wishlist, pd.DataFrame([{"owner_id": owner_id, "id": item_id}])],
                ignore_index=True
            )
            self.save_df(self.wishlist, WISHLIST_CSV)

        self.link_var.set("")
        self.refresh_wishlist_tiles()
        self.refresh_recommendations()

    def add_selected_to_wishlist(self):
        idx = self.product_combo.current()
        if idx < 0 or len(self.products) == 0:
            return
        row = self.products.iloc[idx]
        owner_id = int(row["owner_id"])
        item_id = int(row["id"])

        already = ((self.wishlist["owner_id"] == owner_id) & (self.wishlist["id"] == item_id)).any()
        if not already:
            self.wishlist = pd.concat(
                [self.wishlist, pd.DataFrame([{"owner_id": owner_id, "id": item_id}])],
                ignore_index=True
            )
            self.save_df(self.wishlist, WISHLIST_CSV)

        self.refresh_wishlist_tiles()
        self.refresh_recommendations()

    def remove_from_wishlist(self, owner_id: int, item_id: int):
        self.wishlist = self.wishlist[~((self.wishlist["owner_id"] == owner_id) & (self.wishlist["id"] == item_id))]
        self.save_df(self.wishlist, WISHLIST_CSV)
        self.refresh_wishlist_tiles()
        self.refresh_recommendations()

    def _clear_wishlist_tiles(self):
        for child in self.wl_inner.winfo_children():
            child.destroy()

    def refresh_wishlist_tiles(self):
        self._clear_wishlist_tiles()

        if self.wishlist.empty:
            ttk.Label(self.wl_inner, text="Wish List пуст. Добавь товары по ссылке или из каталога.").pack(
                anchor="w", padx=10, pady=10
            )
            return

        wl = self.wishlist.merge(self.products, on=["owner_id", "id"], how="left")
        wl["title"] = wl["title"].fillna("")
        wl["description"] = wl["description"].fillna("")
        wl["photo_url"] = wl["photo_url"].fillna("")
        wl["url"] = wl["url"].fillna("")

        cols = 3
        pad = 10

        for i, r in enumerate(wl.itertuples()):
            row = i // cols
            col = i % cols

            owner_id = int(r.owner_id)
            item_id = int(r.id)

            # highlight by feedback if exists
            fb = self.get_feedback_label(owner_id, item_id)
            bg = "#ffffff"
            if fb == 1:
                bg = "#e8f7ea"   # light green
            elif fb == 0:
                bg = "#fdeaea"   # light red

            card = tk.Frame(self.wl_inner, bg=bg, bd=1, relief="solid")
            card.grid(row=row, column=col, padx=pad, pady=pad, sticky="nsew")
            self.wl_inner.grid_columnconfigure(col, weight=1)

            title = (r.title or "").strip()
            if not title:
                title = f"(без названия) market{owner_id}_{item_id}"

            # image
            img = self.load_image_thumb(r.photo_url, size=(160, 160))
            if img is None:
                tk.Label(card, text="(нет фото)", bg=bg, width=24, height=10).pack(pady=(8, 0))
            else:
                lbl = tk.Label(card, image=img, bg=bg)
                lbl.image = img
                lbl.pack(pady=(8, 0))

            tk.Label(card, text=title, bg=bg, wraplength=300, justify="left").pack(anchor="w", padx=10, pady=(8, 2))

            price_txt = ""
            if r.price is not None and not (isinstance(r.price, float) and np.isnan(r.price)):
                price_txt = f"Цена: {r.price}"
            tk.Label(card, text=price_txt, bg=bg, fg="#333").pack(anchor="w", padx=10)

            desc = self._short(r.description, 90)
            tk.Label(card, text=desc, bg=bg, fg="#555", wraplength=300, justify="left").pack(
                anchor="w", padx=10, pady=(4, 6)
            )

            fb_text = "Твоя оценка: —"
            if fb == 1:
                fb_text = "Твоя оценка: 👍"
            elif fb == 0:
                fb_text = "Твоя оценка: 👎"
            tk.Label(card, text=fb_text, bg=bg, fg="#444").pack(anchor="w", padx=10, pady=(0, 6))

            btns = tk.Frame(card, bg=bg)
            btns.pack(fill="x", padx=10, pady=(0, 10))

            ttk.Button(
                btns, text="👍",
                command=lambda oid=owner_id, iid=item_id: self._rate_refresh_everywhere(oid, iid, 1)
            ).pack(side="left")

            ttk.Button(
                btns, text="👎",
                command=lambda oid=owner_id, iid=item_id: self._rate_refresh_everywhere(oid, iid, 0)
            ).pack(side="left", padx=6)

            ttk.Button(
                btns, text="Удалить",
                command=lambda oid=owner_id, iid=item_id: self.remove_from_wishlist(oid, iid)
            ).pack(side="right")

    # ---------- Tab 2: Recommendations (TILES) ----------
    def _build_recommend_tab(self):
        top = ttk.Frame(self.tab_rec)
        top.pack(fill="x", padx=10, pady=10)

        self.status_lbl = ttk.Label(top, text="Статус: готово")
        self.status_lbl.pack(anchor="w")

        # Scrollable area for tiles
        self.rec_canvas = tk.Canvas(self.tab_rec, highlightthickness=0)
        self.rec_scroll = ttk.Scrollbar(self.tab_rec, orient="vertical", command=self.rec_canvas.yview)
        self.rec_canvas.configure(yscrollcommand=self.rec_scroll.set)
        self.rec_scroll.pack(side="right", fill="y")
        self.rec_canvas.pack(side="left", fill="both", expand=True, padx=10, pady=10)

        self.rec_inner = ttk.Frame(self.rec_canvas)
        self.rec_window = self.rec_canvas.create_window((0, 0), window=self.rec_inner, anchor="nw")
        self.rec_inner.bind("<Configure>", lambda e: self.rec_canvas.configure(scrollregion=self.rec_canvas.bbox("all")))
        self.rec_canvas.bind("<Configure>", lambda e: self.rec_canvas.itemconfig(self.rec_window, width=e.width))

        btnbar = ttk.Frame(top)
        btnbar.pack(fill="x", pady=6)
        ttk.Button(btnbar, text="Обновить рекомендации", command=self.refresh_recommendations).pack(side="right")

        ttk.Label(
            top,
            text="Подсказка: после ≥6 оценок и наличия и 👍 и 👎 включается обучаемая модель (LogisticRegression).",
            foreground="#555"
        ).pack(anchor="w", pady=4)

    def _clear_recommend_tiles(self):
        for child in self.rec_inner.winfo_children():
            child.destroy()

    def _rate_refresh_everywhere(self, owner_id: int, item_id: int, label: int):
        self.set_feedback(owner_id, item_id, label)
        self.refresh_wishlist_tiles()
        self.refresh_recommendations()

    def _add_to_wishlist_and_refresh(self, owner_id: int, item_id: int):
        already = ((self.wishlist["owner_id"] == owner_id) & (self.wishlist["id"] == item_id)).any()
        if not already:
            self.wishlist = pd.concat(
                [self.wishlist, pd.DataFrame([{"owner_id": owner_id, "id": item_id}])],
                ignore_index=True
            )
            self.save_df(self.wishlist, WISHLIST_CSV)
        self.refresh_wishlist_tiles()
        self.refresh_recommendations()

    def refresh_recommendations(self):
        self._clear_recommend_tiles()

        if self.X is None or len(self.products) == 0:
            self.status_lbl.config(text="Статус: каталог пуст — добавь товары")
            ttk.Label(self.rec_inner, text="Каталог пуст. Добавь товары по ссылке/из источников.").pack(
                anchor="w", padx=10, pady=10
            )
            return

        if self.wishlist.empty:
            self.status_lbl.config(text="Статус: добавь товары в Wish List")
            ttk.Label(self.rec_inner, text="Wish List пуст. Добавь товары — тогда появятся рекомендации.").pack(
                anchor="w", padx=10, pady=10
            )
            return

        # find indices of wishlist items in products
        wl_prod_idx = []
        for w in self.wishlist.itertuples():
            match = self.products.index[
                (self.products["owner_id"] == int(w.owner_id)) & (self.products["id"] == int(w.id))
            ]
            if len(match) > 0:
                wl_prod_idx.append(int(match[0]))

        if not wl_prod_idx:
            self.status_lbl.config(text="Статус: товары из Wish List не найдены в каталоге")
            ttk.Label(self.rec_inner, text="Не удалось сопоставить Wish List с каталогом.").pack(
                anchor="w", padx=10, pady=10
            )
            return

        # защита: если по какой-то причине индексы выходят за границы X
        wl_prod_idx = [i for i in wl_prod_idx if 0 <= i < self.X.shape[0]]
        if not wl_prod_idx:
            self.status_lbl.config(text="Статус: Wish List не совпал с текущим каталогом")
            ttk.Label(self.rec_inner, text="Товары из Wish List не найдены в текущем каталоге.").pack(
                anchor="w", padx=10, pady=10
            )
            return

        # content similarity
        user_vec = self.X[wl_prod_idx].mean(axis=0)
        user_vec = np.asarray(user_vec)
        sims = cosine_similarity(user_vec, self.X).ravel()

        rec_df = self.products.copy()
        rec_df["sim"] = sims
        rec_df = rec_df[~rec_df.index.isin(wl_prod_idx)].copy()

        # ML re-rank if enough feedback
        feedback_join = self.feedback.merge(self.products, on=["owner_id", "id"], how="inner")
        use_model = len(feedback_join) >= 6 and feedback_join["label"].nunique() > 1

        if use_model:
            fb_idx = []
            for f in self.feedback.itertuples():
                match = self.products.index[
                    (self.products["owner_id"] == int(f.owner_id)) & (self.products["id"] == int(f.id))
                ]
                if len(match) > 0:
                    fb_idx.append(int(match[0]))

            X_fb = self.X[fb_idx]
            y_fb = self.feedback["label"].astype(int).values

            model = LogisticRegression(max_iter=1000)
            model.fit(X_fb, y_fb)

            proba = model.predict_proba(self.X)[:, 1]
            rec_df["ml"] = proba[rec_df.index]
            rec_df["score"] = 0.6 * rec_df["sim"] + 0.4 * rec_df["ml"]
            self.status_lbl.config(text=f"Статус: модель обучена на {len(feedback_join)} оценках ✅")
        else:
            rec_df["score"] = rec_df["sim"]
            self.status_lbl.config(text="Статус: мало оценок для обучения (нужно ≥6 и разные классы)")

        top = rec_df.sort_values("score", ascending=False).head(24)

        cols = 3
        pad = 10

        for i, r in enumerate(top.itertuples()):
            row = i // cols
            col = i % cols

            owner_id = int(r.owner_id)
            item_id = int(r.id)

            fb = self.get_feedback_label(owner_id, item_id)
            bg = "#ffffff"
            if fb == 1:
                bg = "#e8f7ea"   # liked
            elif fb == 0:
                bg = "#fdeaea"   # disliked

            card = tk.Frame(self.rec_inner, bg=bg, bd=1, relief="solid")
            card.grid(row=row, column=col, padx=pad, pady=pad, sticky="nsew")
            self.rec_inner.grid_columnconfigure(col, weight=1)

            title = (r.title or "").strip()
            if not title:
                title = f"(без названия) market{owner_id}_{item_id}"

            img = self.load_image_thumb(r.photo_url, size=(160, 160))
            if img is None:
                tk.Label(card, text="(нет фото)", bg=bg, width=24, height=10).pack(pady=(8, 0))
            else:
                lbl = tk.Label(card, image=img, bg=bg)
                lbl.image = img
                lbl.pack(pady=(8, 0))

            tk.Label(card, text=title, bg=bg, wraplength=300, justify="left").pack(anchor="w", padx=10, pady=(8, 2))

            price_txt = ""
            if r.price is not None and not (isinstance(r.price, float) and np.isnan(r.price)):
                price_txt = f"Цена: {r.price}"
            tk.Label(card, text=price_txt, bg=bg, fg="#333").pack(anchor="w", padx=10)

            desc = self._short(r.description, 90)
            tk.Label(card, text=desc, bg=bg, fg="#555", wraplength=300, justify="left").pack(
                anchor="w", padx=10, pady=(4, 6)
            )

            fb_text = "Твоя оценка: —"
            if fb == 1:
                fb_text = "Твоя оценка: 👍"
            elif fb == 0:
                fb_text = "Твоя оценка: 👎"
            tk.Label(card, text=fb_text, bg=bg, fg="#444").pack(anchor="w", padx=10, pady=(0, 6))

            tk.Label(card, text=f"score={float(r.score):.3f}", bg=bg, fg="#777").pack(anchor="w", padx=10, pady=(0, 6))

            btns = tk.Frame(card, bg=bg)
            btns.pack(fill="x", padx=10, pady=(0, 10))

            ttk.Button(
                btns, text="👍 Like",
                command=lambda oid=owner_id, iid=item_id: self._rate_refresh_everywhere(oid, iid, 1)
            ).pack(side="left")

            ttk.Button(
                btns, text="👎 Dislike",
                command=lambda oid=owner_id, iid=item_id: self._rate_refresh_everywhere(oid, iid, 0)
            ).pack(side="left", padx=8)

            ttk.Button(
                btns, text="➕ в Wish List",
                command=lambda oid=owner_id, iid=item_id: self._add_to_wishlist_and_refresh(oid, iid)
            ).pack(side="right")


if __name__ == "__main__":
    app = WishRecommenderApp()
    app.mainloop()
