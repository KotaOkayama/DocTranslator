# app/utils/favicon_generator.py
from PIL import Image, ImageDraw, ImageFont
import os
import logging

logger = logging.getLogger("doctranslator")


def generate_favicon(text="DT", size=256, output_dir=None):
    """
    指定されたテキストでファビコンを生成

    Args:
        text (str): ファビコンに表示するテキスト
        size (int): ファビコンのサイズ（ピクセル）
        output_dir (str): 出力ディレクトリ
    """
    try:
        if output_dir is None:
            # デフォルトの出力ディレクトリ
            output_dir = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), "static"
            )

        # カラーパレット
        background_color = (0, 123, 255)  # Bootstrap primary blue
        text_color = (255, 255, 255)  # 白

        # 新しい画像を作成
        image = Image.new("RGBA", (size, size), background_color)
        draw = ImageDraw.Draw(image)

        # フォントの設定（システムフォントを使用）
        try:
            # macOS用
            font_path = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
            if not os.path.exists(font_path):
                # macOS用の別のパス
                font_path = "/System/Library/Fonts/Supplemental/Arial.ttf"

            if not os.path.exists(font_path):
                # Linuxやその他のシステム用のフォールバック
                font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

            if not os.path.exists(font_path):
                # Windowsのフォント
                font_path = "C:\\Windows\\Fonts\\Arial.ttf"

            font = ImageFont.truetype(font_path, size=int(size * 0.6))
        except IOError:
            # フォントが見つからない場合はデフォルトフォントを使用
            font = ImageFont.load_default()

        # テキストのサイズを計算
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        # テキストを中央に配置
        position = ((size - text_width) / 2, (size - text_height) / 2)

        # テキストを描画
        draw.text(position, text, font=font, fill=text_color)

        # 出力ディレクトリを作成（存在しない場合）
        os.makedirs(output_dir, exist_ok=True)

        # 様々なサイズのファビコンを生成
        favicon_sizes = [16, 32, 48, 64, 128, 256]

        for favicon_size in favicon_sizes:
            # リサイズ
            resized_image = image.resize((favicon_size, favicon_size), Image.LANCZOS)

            # ICO形式で保存
            if favicon_size == 16:
                resized_image.save(
                    os.path.join(output_dir, "favicon.ico"),
                    format="ICO",
                    sizes=[(16, 16)],
                )

            # PNG形式でも保存
            resized_image.save(
                os.path.join(output_dir, f"favicon-{favicon_size}.png"), format="PNG"
            )

        logger.info(f"Faviconを生成しました: {output_dir}")
        return True
    except Exception as e:
        logger.error(f"Favicon生成エラー: {e}")
        return False
