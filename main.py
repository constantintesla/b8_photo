import os
import shutil
import cv2
import numpy as np
from PIL import Image, ImageTk
from ultralytics import YOLO
import tkinter as tk
from tkinter import messagebox, filedialog  # Добавляем импорт для messagebox и filedialog


def safe_image_open(path):
    try:
        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is not None:
            return img
        with Image.open(path) as pil_img:
            return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"⚠️ Ошибка открытия {path}: {str(e)}")
        return None


def detect_faces(image_path):
    try:
        # Загрузка модели YOLOv11 (или другой версии)
        model = YOLO("yolov11n-face.pt")  # Убедитесь, что файл модели доступен
        results = model(image_path)  # Выполняем предсказание

        faces = []
        confidence_threshold = 0.3  # Порог уверенности (можно настроить)

        # Обработка результатов
        for result in results:
            boxes = result.boxes.cpu().numpy()  # Получаем bounding boxes
            for box in boxes:
                if box.conf[0] > confidence_threshold:  # Проверяем уверенность
                    x1, y1, x2, y2 = map(int, box.xyxy[0])  # Координаты прямоугольника
                    w, h = x2 - x1, y2 - y1
                    faces.append([x1, y1, w, h])

        return np.array(faces) if len(faces) > 0 else np.empty((0, 4))
    except Exception as e:
        print(f"⚠️ Ошибка детекции: {str(e)}")
        return np.empty((0, 4))  # Возвращаем пустой массив при ошибке


def crop_to_passport(image_path, output_path, face_coords):
    try:
        img = Image.open(image_path)
        x, y, w, h = face_coords

        target_w, target_h = 412, 531
        target_ratio = target_w / target_h

        # Рассчитываем область обрезки с сохранением пропорций
        crop_height = int(h * 1.4)  # Голова + плечи
        crop_width = int(crop_height * target_ratio)

        # Центрируем
        x_center = x + w // 2
        y_center = y + h // 2

        x1 = max(0, x_center - crop_width // 2)
        y1 = max(0, y_center - h // 2 - int(h * 0.1))  # 10% сверху
        x2 = min(img.width, x1 + crop_width)
        y2 = min(img.height, y1 + crop_height)

        # Корректировка если вышли за границы
        if x2 - x1 < crop_width:
            x1 = max(0, x2 - crop_width)
        if y2 - y1 < crop_height:
            y1 = max(0, y2 - crop_height)

        print(f"✂️ Обрезка области: ({x1}, {y1}, {x2}, {y2})")

        cropped = img.crop((x1, y1, x2, y2))
        cropped = cropped.resize((target_w, target_h), Image.Resampling.LANCZOS)
        cropped.save(output_path, quality=95)

    except Exception as e:
        print(f"⚠️ Ошибка обрезки: {str(e)}")
        shutil.copy2(image_path, output_path)


def process_folder(src_dir, dst_dir):
    for root, _, files in os.walk(src_dir):
        rel_path = os.path.relpath(root, src_dir)
        dst_path = os.path.join(dst_dir, rel_path)
        os.makedirs(dst_path, exist_ok=True)

        for file in [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]:
            src_file = os.path.join(root, file)
            dst_file = os.path.join(dst_path, file)

            if os.path.exists(dst_file):
                print(f"⚠️ Файл уже существует: {dst_file}")
                continue

            faces = detect_faces(src_file)
            print(f"🔍 Обнаружено лиц: {len(faces)} для файла: {src_file}")

            if len(faces) > 0:  # Проверяем длину массива
                print(f"✂️ Обрезка изображения: {src_file}")
                try:
                    # Предварительная обрезка
                    crop_to_passport(src_file, dst_file, faces[0])  # Берем первое обнаруженное лицо
                except Exception as e:
                    print(f"❌ Ошибка обрезки: {str(e)}")
                    shutil.copy2(src_file, dst_file)
            else:
                print(f"⚠️ Лицо не обнаружено, копирование: {src_file}")
                shutil.copy2(src_file, dst_file)


def create_gui():
    def start_processing():
        source = source_entry.get()
        output = output_entry.get()

        if not os.path.isdir(source):
            messagebox.showerror("Ошибка", "Исходная папка не существует!")
            return

        if not os.path.isdir(output):
            os.makedirs(output, exist_ok=True)

        print("🔄 Начало обработки...")
        process_folder(source, output)
        print(f"✅ Готово! Результаты в: {output}")
        messagebox.showinfo("Готово", f"Обработка завершена! Результаты в: {output}")

    # Создаем главное окно
    root = tk.Tk()
    root.title("Обработка изображений")

    # Поле для ввода исходной папки
    tk.Label(root, text="Исходная папка:").grid(row=0, column=0, padx=10, pady=5, sticky="w")
    source_entry = tk.Entry(root, width=50)
    source_entry.grid(row=0, column=1, padx=10, pady=5)
    source_entry.insert(0, r"Фото")  # Значение по умолчанию

    # Поле для ввода папки назначения
    tk.Label(root, text="Папка назначения:").grid(row=1, column=0, padx=10, pady=5, sticky="w")
    output_entry = tk.Entry(root, width=50)
    output_entry.grid(row=1, column=1, padx=10, pady=5)
    output_entry.insert(0, r"Результат")  # Значение по умолчанию

    # Кнопка для выбора исходной папки
    def select_source_folder():
        folder = filedialog.askdirectory()
        if folder:
            source_entry.delete(0, tk.END)
            source_entry.insert(0, folder)

    tk.Button(root, text="Выбрать", command=select_source_folder).grid(row=0, column=2, padx=5, pady=5)

    # Кнопка для выбора папки назначения
    def select_output_folder():
        folder = filedialog.askdirectory()
        if folder:
            output_entry.delete(0, tk.END)
            output_entry.insert(0, folder)

    tk.Button(root, text="Выбрать", command=select_output_folder).grid(row=1, column=2, padx=5, pady=5)

    # Кнопка для запуска обработки
    tk.Button(root, text="Обработать все файлы", command=start_processing).grid(
        row=2, column=0, columnspan=3, pady=10
    )

    root.mainloop()


if __name__ == "__main__":
    create_gui()