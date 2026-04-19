import os
import sys
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F
from PIL import Image, ImageTk, ImageDraw, ImageFont
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox

NUM_CLASSES = 7 
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
SCORE_THRESHOLD = 0.3
NMS_THRESHOLD = 0.4

CHECKPOINT = os.path.join("checkpoints", "best_model.pth")

DEFECT_NAMES = {
    1: "mouse_bite",
    2: "spur",
    3: "missing_hole",
    4: "short",
    5: "open_circuit",
    6: "spurious_copper",
}

DEFECT_COLORS = {
    1: (255, 0, 0),    
    2: (255, 165, 0),   
    3: (255, 255, 0),     
    4: (0, 0, 255),       
    5: (255, 0, 255),    
    6: (0, 255, 0),       
}


def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=None,
        weights_backbone=None,
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def load_model():
    model = get_model(NUM_CLASSES)

    ckpt_path = CHECKPOINT
    if not os.path.exists(ckpt_path):
        msg = f"Nije pronadjen checkpoint"
        messagebox.showerror("Greska", msg)
        sys.exit(1)

    print(f"Ucitavam model iz: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        epoch = checkpoint.get("epoch", "?")
        mAP = checkpoint.get("val_mAP", "?")
        print(f"  Epoch: {epoch}, mAP: {mAP}")
    else:
        model.load_state_dict(checkpoint)

    model.to(DEVICE)
    model.eval()
    model.roi_heads.score_thresh = SCORE_THRESHOLD
    model.roi_heads.nms_thresh = NMS_THRESHOLD

    print(f"Model ucitan)")
    return model


def detect(model, image_path):
    img = Image.open(image_path).convert("RGB")
    img_tensor = F.to_tensor(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(img_tensor)[0]

    keep = outputs["scores"] >= SCORE_THRESHOLD
    boxes = outputs["boxes"][keep].cpu().numpy()
    labels = outputs["labels"][keep].cpu().numpy()
    scores = outputs["scores"][keep].cpu().numpy()

    return boxes, labels, scores


def draw_detections_pil(image_path, boxes, labels, scores):
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("arial.ttf", 14)
        font_small = ImageFont.truetype("arial.ttf", 12)
    except (IOError, OSError):
        font = ImageFont.load_default()
        font_small = font

    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box.astype(int)
        color = DEFECT_COLORS.get(label, (255, 255, 255))
        name = DEFECT_NAMES.get(label, f"class_{label}")

        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

        text = f"{name}: {score:.2f}"
        bbox = draw.textbbox((x1, y1 - 18), text, font=font_small)
        draw.rectangle([bbox[0] - 1, bbox[1] - 1, bbox[2] + 1, bbox[3] + 1], fill=color)
        draw.text((x1, y1 - 18), text, fill=(0, 0, 0), font=font_small)

    info = f"Detektovano defekata: {len(boxes)}"
    draw.text((10, 8), info, fill=(0, 255, 0), font=font)

    return img


class PCBDetectorApp:

    def __init__(self, root, model):
        self.root = root
        self.model = model
        self.current_image_path = None
        self.result_img = None

        self.root.title("PCB Defect Detector")
        self.root.configure(bg="#2b2b2b")
        self.root.resizable(True, True)

        btn_frame = tk.Frame(root, bg="#3c3f41", pady=8, padx=10)
        btn_frame.pack(side=tk.TOP, fill=tk.X)

        btn_style = {
            "font": ("Segoe UI", 11),
            "padx": 16, "pady": 6,
            "cursor": "hand2",
            "relief": "flat",
            "bd": 0,
        }

        self.btn_open = tk.Button(
            btn_frame, text="\U0001F4C2  Izaberi sliku",
            bg="#4CAF50", fg="white", activebackground="#45a049",
            command=self.on_open, **btn_style
        )
        self.btn_open.pack(side=tk.LEFT, padx=(0, 8))

        self.btn_save = tk.Button(
            btn_frame, text="\U0001F4BE  Sacuvaj rezultat",
            bg="#2196F3", fg="white", activebackground="#1e88e5",
            command=self.on_save, state=tk.DISABLED, **btn_style
        )
        self.btn_save.pack(side=tk.LEFT, padx=(0, 8))

        self.btn_exit = tk.Button(
            btn_frame, text="\u274C  Izlaz",
            bg="#f44336", fg="white", activebackground="#e53935",
            command=self.on_exit, **btn_style
        )
        self.btn_exit.pack(side=tk.RIGHT)

        self.status_var = tk.StringVar(value="Ucitajte sliku za analizu")
        self.status_label = tk.Label(
            btn_frame, textvariable=self.status_var,
            bg="#3c3f41", fg="#aaaaaa", font=("Segoe UI", 10),
            anchor="w"
        )
        self.status_label.pack(side=tk.LEFT, padx=15, fill=tk.X, expand=True)

        self.canvas_frame = tk.Frame(root, bg="#2b2b2b")
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(self.canvas_frame, bg="#1e1e1e", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.canvas.create_text(
            400, 300, text="Kliknite 'Izaberi sliku' da pocnete",
            fill="#666666", font=("Segoe UI", 16), tags="placeholder"
        )

        self.results_frame = tk.Frame(root, bg="#3c3f41", pady=5, padx=10)
        self.results_frame.pack(side=tk.BOTTOM, fill=tk.X)

        self.results_var = tk.StringVar(value="")
        self.results_label = tk.Label(
            self.results_frame, textvariable=self.results_var,
            bg="#3c3f41", fg="#cccccc", font=("Consolas", 10),
            anchor="w", justify=tk.LEFT
        )
        self.results_label.pack(fill=tk.X)


        self.root.geometry("850x650")
        self.root.minsize(600, 450)

        # Bind resize
        self.canvas.bind("<Configure>", self._on_canvas_resize)
        self._photo_ref = None 

    def on_open(self):
        file_path = filedialog.askopenfilename(
            title="Izaberite PCB sliku za analizu",
            filetypes=[
                ("Slike", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif"),
                ("JPEG", "*.jpg *.jpeg"),
                ("PNG", "*.png"),
                ("Svi fajlovi", "*.*"),
            ],
        )

        if not file_path:
            return

        if not os.path.exists(file_path):
            messagebox.showerror("Greska", f"Fajl ne postoji: {file_path}")
            return

        self.current_image_path = file_path
        self.status_var.set(f"Analiza: {os.path.basename(file_path)}...")
        self.root.update_idletasks()

        boxes, labels, scores = detect(self.model, file_path)
        self.result_img = draw_detections_pil(file_path, boxes, labels, scores)
        self._display_image()
        self._update_results(file_path, boxes, labels, scores)
        self.btn_save.config(state=tk.NORMAL)
        self._print_results(file_path, boxes, labels, scores)

    def _display_image(self):
        if self.result_img is None:
            return

        self.canvas.delete("all")

        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw < 10 or ch < 10:
            cw, ch = 800, 550

        iw, ih = self.result_img.size
        scale = min(cw / iw, ch / ih, 1.0)
        new_w = int(iw * scale)
        new_h = int(ih * scale)

        display_img = self.result_img.resize((new_w, new_h), Image.LANCZOS)
        self._photo_ref = ImageTk.PhotoImage(display_img)

        x = cw // 2
        y = ch // 2
        self.canvas.create_image(x, y, image=self._photo_ref, anchor=tk.CENTER)

    def _on_canvas_resize(self, event):
        if self.result_img is not None:
            self._display_image()

    def _update_results(self, image_path, boxes, labels, scores):
        filename = os.path.basename(image_path)
        if len(boxes) == 0:
            text = f"{filename}  |  Nema detektovanih defekata"
        else:
            defects = []
            for label, score in zip(labels, scores):
                name = DEFECT_NAMES.get(label, f"class_{label}")
                defects.append(f"{name}({score:.2f})")
            text = f"{filename}  |  {len(boxes)} defekata:  {',  '.join(defects)}"

        self.results_var.set(text)
        self.status_var.set(f"Zavrseno: {filename}")

    def _print_results(self, image_path, boxes, labels, scores):
        print(f"\n{'='*50}")
        print(f"Slika: {os.path.basename(image_path)}")
        print(f"Broj detekcija: {len(boxes)}")
        print(f"{'-'*50}")
        if len(boxes) == 0:
            print("  Nema detektovanih defekata.")
        else:
            for i, (box, label, score) in enumerate(zip(boxes, labels, scores)):
                name = DEFECT_NAMES.get(label, f"class_{label}")
                x1, y1, x2, y2 = box.astype(int)
                print(f"  [{i+1}] {name:20s}  score={score:.3f}  box=({x1},{y1},{x2},{y2})")
        print(f"{'='*50}")

    def on_save(self):
        if self.result_img is None or self.current_image_path is None:
            return

        save_path = filedialog.asksaveasfilename(
            title="Sacuvaj rezultat",
            initialfile=os.path.splitext(os.path.basename(self.current_image_path))[0] + "_detected.jpg",
            defaultextension=".jpg",
            filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png"), ("Svi fajlovi", "*.*")],
        )

        if save_path:
            self.result_img.save(save_path)
            self.status_var.set(f"Sacuvano: {os.path.basename(save_path)}")
            print(f"Sacuvano: {save_path}")

    def on_exit(self):
        self.root.quit()
        self.root.destroy()


def main():
    model = load_model()

    root = tk.Tk()
    app = PCBDetectorApp(root, model)
    root.mainloop()

if __name__ == "__main__":
    main()
