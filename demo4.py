import warnings
import pandas as pd
from matplotlib import pyplot as plt
warnings.filterwarnings('ignore')
from math import ceil
from model import CNN_Model
from collections import defaultdict
import cv2
import imutils

def filter_unique_contours(contours, overlap_threshold=0.5):
    """
    Lọc các contours để loại bỏ những contours trùng nhau.

    Parameters:
    - contours: Danh sách các contours (x, y, w, h).
    - overlap_threshold: Ngưỡng chồng lấn để định nghĩa là trùng (0.0 - 1.0).

    Returns:
    - unique_contours: Danh sách các contours không trùng.
    """
    unique_contours = []

    for i, (x1, y1, w1, h1) in enumerate(contours):
        keep = True
        for x2, y2, w2, h2 in unique_contours:
            # Tính toán phần diện tích chồng lấn
            x_overlap = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
            y_overlap = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
            overlap_area = x_overlap * y_overlap

            # Tính diện tích nhỏ hơn giữa hai bounding boxes
            box1_area = w1 * h1
            box2_area = w2 * h2
            min_area = min(box1_area, box2_area)

            # Kiểm tra nếu phần diện tích chồng lấn vượt ngưỡng
            if overlap_area / min_area > overlap_threshold:
                keep = False
                break

        if keep:
            unique_contours.append((x1, y1, w1, h1))
    return unique_contours

def crop_image(img, min_width_sbd=120, max_width_sbd=130, min_height_sbd=290, max_height_sbd=310,
               min_width_mdt=60, max_width_mdt=70, min_height_mdt=290, max_height_mdt=310,
               min_width=190, max_width=210, min_height=900, max_height=1050):
    """
    Cắt các contours không trùng lặp từ ảnh gốc và trả về danh sách ảnh grayscale cùng với tọa độ bounding boxes.
    Lưu ảnh cho các loại SBD và MDT.
    """
    # Chuyển ảnh sang trắng đen để phát hiện viền
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Khử nhiễu
    blurred = cv2.GaussianBlur(gray_img, (5, 5), 0)

    # Áp dụng Canny để phát hiện cạnh
    img_canny = cv2.Canny(blurred, 50, 150)

    # Tìm contours
    cnts = cv2.findContours(img_canny.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # Lọc các contours theo kích thước và phân loại thành 3 loại (sbd, mdt, và loại khác)
    contours_sbd = [
        cv2.boundingRect(c) for c in cnts
        if min_width_sbd <= cv2.boundingRect(c)[2] <= max_width_sbd and min_height_sbd <= cv2.boundingRect(c)[3] <= max_height_sbd
    ]
    contours_mdt = [
        cv2.boundingRect(c) for c in cnts
        if min_width_mdt <= cv2.boundingRect(c)[2] <= max_width_mdt and min_height_mdt <= cv2.boundingRect(c)[3] <= max_height_mdt
    ]
    contours_general = [
        cv2.boundingRect(c) for c in cnts
        if min_width <= cv2.boundingRect(c)[2] <= max_width and min_height <= cv2.boundingRect(c)[3] <= max_height
    ]

    # Lọc các contours không trùng lặp
    unique_contours_sbd = contours_sbd   #filter_unique_contours(contours_sbd)
    unique_contours_mdt = contours_mdt #filter_unique_contours(contours_mdt)
    unique_contours_general = filter_unique_contours(contours_general)

    # Chuẩn bị kết quả
    result_general = []
    for x, y, w, h in unique_contours_general:
        contour_img = gray_img[y:y + h, x:x + w]
        result_general.append((contour_img, [x, y, w, h]))

    # Lưu ảnh cho SBD và MDT
    result_sbd = []
    for x, y, w, h in unique_contours_sbd:
        contour_img = gray_img[y:y + h, x:x + w]
        result_sbd.append((contour_img, [x, y, w, h]))
        cv2.imwrite('sbd.png', contour_img)  # Lưu ảnh SBD

    result_mdt = []
    for x, y, w, h in unique_contours_mdt:
        contour_img = gray_img[y:y + h, x:x + w]
        result_mdt.append((contour_img, [x, y, w, h]))
        cv2.imwrite('mdt.png', contour_img)  # Lưu ảnh MDT

    # Sắp xếp danh sách theo tọa độ x
    result_general = sorted(result_general, key=lambda item: item[1][0])

    # Trả về danh sách kết quả chung
    return result_general



def process_ans_blocks(ans_blocks):
    """
    this function processes 2 block answer box and returns a list of answers with a length of 200 bubble choices
    :param ans_blocks: a list that includes 2 elements, each element has the format of [image, [x, y, w, h]]
    """
    list_answers = []

    # Loop over each block in ans_blocks
    for ans_block in ans_blocks:
        ans_block_img = np.array(ans_block[0])

        # Ensure the image has a valid shape
        #print("Shape of ans_block_img:", ans_block_img.shape)
        if ans_block_img.shape[0] < 6:
            raise ValueError(f"Image height is too small: {ans_block_img.shape[0]}")

        offset1 = ceil(ans_block_img.shape[0] / 6)

        # Loop over each box in the answer block
        for i in range(6):
            box_img = np.array(ans_block_img[i * offset1:(i + 1) * offset1, :])
            height_box = box_img.shape[0]

            box_img = box_img[14:height_box - 14, :]
            offset2 = ceil(box_img.shape[0] / 5)

            # Loop over each line in a box
            for j in range(5):
                list_answers.append(box_img[j * offset2:(j + 1) * offset2, :])

    return list_answers


def process_list_ans(list_answers):
    list_choices = []
    offset = 44
    start = 32

    for answer_img in list_answers:
        for i in range(4):
            bubble_choice = answer_img[:, start + i * offset:start + (i + 1) * offset]
            bubble_choice = cv2.threshold(bubble_choice, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

            bubble_choice = cv2.resize(bubble_choice, (28, 28), cv2.INTER_AREA)
            bubble_choice = bubble_choice.reshape((28, 28, 1))
            list_choices.append(bubble_choice)

    if len(list_choices) != 480:
        print(len(list_choices))
        raise ValueError("Length of list_choices must be 480")
    return list_choices


def map_answer(idx):
    if idx % 4 == 0:
        answer_circle = "A"
    elif idx % 4 == 1:
        answer_circle = "B"
    elif idx % 4 == 2:
        answer_circle = "C"
    else:
        answer_circle = "D"
    return answer_circle


def get_answers(list_answers):
    results = defaultdict(list)
    model = CNN_Model('weight.h5').build_model(rt=True)
    list_answers = np.array(list_answers)
    scores = model.predict_on_batch(list_answers / 255.0)
    for idx, score in enumerate(scores):
        question = idx // 4

        # score [unchoiced_cf, choiced_cf]
        if score[1] > 0.9:  # choiced confidence score > 0.9
            chosed_answer = map_answer(idx)
            results[question + 1].append(chosed_answer)

    return results


import os
import numpy as np

def saveimg(images_array, output_dir):
    # Kiểm tra nếu thư mục không tồn tại thì tạo mới
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Duyệt qua từng ảnh trong mảng
    for i, img_array in enumerate(images_array):
        # Kiểm tra kiểu dữ liệu của mảng
        # Tạo thư mục nếu chưa tồn tại
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for idx, image_data in enumerate(list_ans):
            plt.imshow(image_data, cmap='gray')  # Hiển thị ảnh grayscale
            plt.axis('off')  # Ẩn trục tọa độ

            # Lưu ảnh với tên file index
            file_path = os.path.join(output_dir, f"image_{idx + 1}.png")
            plt.savefig(file_path, bbox_inches='tight', pad_inches=0)
            plt.close()  # Đóng plot để tránh lưu trữ không cần thiết

        print(f"Tất cả ảnh đã được lưu tại {output_dir}")
        break
def split_and_process_image(image_path, rows, cols, output_dir):
    """
    Cắt hình ảnh thành các ô nhỏ, thay đổi kích thước và xử lý ảnh (nền đen, chữ trắng),
    rồi lưu vào thư mục và trả về danh sách các ô.
    :param image_path: Đường dẫn đến hình ảnh
    :param rows: Số dòng cần chia
    :param cols: Số cột cần chia
    :param output_dir: Thư mục lưu các ô hình ảnh đã xử lý
    :return: Danh sách các ô đã xử lý (mỗi ô là một mảng numpy)
    """
    # Đọc hình ảnh
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Không thể đọc hình ảnh, kiểm tra lại đường dẫn.")

    # Tạo thư mục lưu nếu chưa tồn tại
    os.makedirs(output_dir, exist_ok=True)

    # Lấy kích thước ảnh
    height, width, _ = img.shape

    # Tính kích thước của mỗi ô
    cell_height = height // rows
    cell_width = width // cols

    # Danh sách lưu các ô đã xử lý
    cells = []

    # Cắt ảnh theo dòng và cột, xử lý từng ô
    for i in range(rows):
        for j in range(cols):
            # Xác định tọa độ cắt
            start_y = i * cell_height
            end_y = (i + 1) * cell_height
            start_x = j * cell_width
            end_x = (j + 1) * cell_width

            # Cắt ảnh
            cropped_img = img[start_y:end_y, start_x:end_x]

            # Thay đổi kích thước mỗi ô về (28, 28)
            resized_cell = cv2.resize(cropped_img, (28, 28))

            # Chuyển ô thành ảnh đen trắng (grayscale)
            gray_cell = cv2.cvtColor(resized_cell, cv2.COLOR_BGR2GRAY)

            # Đảo ngược ảnh để nền đen và chữ trắng
            inverted_cell = cv2.bitwise_not(gray_cell)

            # Thêm kênh đơn (1) để có kích thước (28, 28, 1)
            inverted_cell = inverted_cell[..., np.newaxis]

            # Lưu ô vào thư mục
            cell_filename = os.path.join(output_dir, f"cell_{i}_{j}.png")
            cv2.imwrite(cell_filename, inverted_cell.squeeze())  # .squeeze() để loại bỏ kênh đơn

            # Thêm ô vào danh sách
            cells.append(inverted_cell)

    # Kiểm tra số lượng ô đã cắt (nên là rows * cols)
    if len(cells) != rows * cols:
        print(len(cells))
        raise ValueError(f"Số lượng ô cắt phải là {rows * cols}.")

    return cells



# Hàm nhận diện với mô hình CNN và xây dựng ma trận 10xN
def recognize_cells_with_cnn(cells, matrix_rows, matrix_cols):
    """
    Dùng mô hình CNN để nhận diện các ảnh cắt, xây dựng ma trận và trả về chuỗi số kết hợp từ các cột của ma trận.
    :param cells: Danh sách các ô hình ảnh (mảng numpy kích thước (28, 28, 1)), tổng cộng matrix_rows * matrix_cols ô.
    :param matrix_rows: Số dòng của ma trận kết quả.
    :param matrix_cols: Số cột của ma trận kết quả.
    :return: Chuỗi số kết hợp từ các cột của ma trận.
    """
    # Tải mô hình CNN đã huấn luyện
    model = CNN_Model('weight.h5').build_model(rt=True)

    # Tạo ma trận có kích thước (matrix_rows, matrix_cols) để lưu các vị trí (ban đầu tất cả là 0)
    matrix = np.zeros((matrix_rows, matrix_cols), dtype=int)

    # Chuyển các ô thành định dạng mà mô hình CNN có thể xử lý
    cells = np.array(cells)

    # Dự đoán với mô hình CNN
    scores = model.predict_on_batch(cells / 255.0)  # Đưa ảnh về phạm vi [0, 1]

    # Xử lý kết quả dự đoán và cập nhật ma trận
    for idx, score in enumerate(scores):
        row = idx // matrix_cols  # Xác định dòng (mỗi dòng có matrix_cols ảnh)
        col = idx % matrix_cols   # Xác định cột (mỗi dòng có matrix_cols ảnh, cột từ 0 đến matrix_cols-1)

        # score [unchoiced_cf, choiced_cf]
        if score[1] > 0.9:  # Nếu confidence cho chọn là tô lớn hơn 0.9
            matrix[row, col] = 1  # Đánh dấu vào ma trận (1 là tô, 0 là không tô)

    # Đọc ma trận theo cột và xây dựng chuỗi số
    final_string = ""
    for col in range(matrix_cols):  # Duyệt qua các cột
        for row in range(matrix_rows):  # Duyệt qua các dòng
            if matrix[row, col] == 1:
                final_string += str(row)  # Nếu ô được tô, thêm vào chuỗi

    return final_string

def score(predicted_answers, ground_truth, output_file):
    total_questions = len(ground_truth)
    correct_answers = 0
    result_data = []

    for question, correct_answers_list in ground_truth.items():
        predicted_list = predicted_answers.get(question, [])

        # So sánh các câu trả lời đúng và câu trả lời dự đoán
        is_correct = 'Đúng' if set(correct_answers_list) == set(predicted_list) else 'Sai'

        if is_correct == 'Đúng':
            correct_answers += 1

        # Lưu dữ liệu câu hỏi, đáp án dự đoán, đáp án đúng và kết quả
        result_data.append([question, ', '.join(predicted_list), ', '.join(correct_answers_list), is_correct])

    # Tính điểm (0.25 / 1 câu)
    result_score = correct_answers * (10 / total_questions)

    # Tạo DataFrame từ dữ liệu kết quả
    df = pd.DataFrame(result_data, columns=['ID Câu', 'Đáp án đã tô', 'Đáp án đúng', 'Kết quả'])

    # Xuất ra file Excel
    df.to_excel(output_file, index=False)

    return result_score

if __name__ == '__main__':
    img = cv2.imread('test4.jpg')
    list_ans_boxes = crop_image(img)
    list_ans = process_ans_blocks(list_ans_boxes)
    list_ans = process_list_ans(list_ans)
    temp1 = split_and_process_image('sbd.png',10, 6,'sbd')
    temp2 = split_and_process_image('mdt.png', 10, 3, 'mdt')
    sbd = recognize_cells_with_cnn(temp1, 10, 6)
    mdt = recognize_cells_with_cnn(temp2, 10, 3)
    print("Số báo danh:", sbd)
    print("Mã đề thi:", mdt)
    answers = get_answers(list_ans)
    #print(answers)
    # Đọc file CSV
    df = pd.read_excel('Dapandethi.xlsx', sheet_name=mdt)
    # Chuẩn bị dữ liệu so sánh
    ground_truth = df.groupby('question_id')['correct_answer'].apply(list).to_dict()
    # Chấm điểm
    result = score(answers, ground_truth, 'result.xlsx')
    print(f"Điểm số:", result)


