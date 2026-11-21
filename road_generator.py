import math
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance

# Dimensions standard for autonomous vehicle camera feed
CAM_WIDTH = 200
CAM_HEIGHT = 66
SIGN_SIZE = 64

# Color palettes
ASPHALT_COLOR = (55, 58, 62)
GRASS_COLOR = (42, 110, 52)
SKY_COLOR = (135, 195, 235)
HORIZON_COLOR = (180, 215, 230)
LANE_WHITE = (235, 235, 235)
LANE_YELLOW = (230, 195, 30)

def draw_polygon_filled(draw, points, color):
    draw.polygon(points, fill=color)

def generate_traffic_sign(sign_id: int, size: int = 64, augment: bool = True) -> Image.Image:
    """
    Renders realistic synthetic traffic sign images.
    sign_id:
      0: Background patch (no sign)
      1: STOP sign
      2: Speed Limit 30
      3: Speed Limit 60
      4: Yield sign
      5: Sharp Turn
    """
    img = Image.new("RGB", (size, size), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    center = size / 2
    r = size * 0.44

    if sign_id == 0:
        # Background noise / asphalt / foliage
        bg_choice = random.choice(["asphalt", "grass", "sky", "mixed"])
        if bg_choice == "asphalt":
            base = np.random.randint(50, 80, (size, size, 3), dtype=np.uint8)
        elif bg_choice == "grass":
            base = np.random.randint(30, 90, (size, size, 3), dtype=np.uint8)
            base[:, :, 1] = np.clip(base[:, :, 1] + 40, 0, 255)
        elif bg_choice == "sky":
            base = np.full((size, size, 3), (150, 200, 240), dtype=np.uint8)
            noise = np.random.randint(-20, 20, (size, size, 3), dtype=np.int16)
            base = np.clip(base + noise, 0, 255).astype(np.uint8)
        else:
            base = np.random.randint(40, 180, (size, size, 3), dtype=np.uint8)
        img = Image.fromarray(base)

    elif sign_id == 1:
        # Background transparent or environmental
        img.paste((180, 195, 205), [0, 0, size, size])
        draw = ImageDraw.Draw(img)
        # Octagon
        angles = [math.radians(22.5 + i * 45) for i in range(8)]
        outer_pts = [(center + r * math.cos(a), center + r * math.sin(a)) for a in angles]
        draw.polygon(outer_pts, fill=(204, 25, 25), outline=(255, 255, 255), width=2)
        inner_r = r * 0.92
        inner_pts = [(center + inner_r * math.cos(a), center + inner_r * math.sin(a)) for a in angles]
        draw.polygon(inner_pts, outline=(255, 255, 255), width=1)
        # STOP text
        draw.text((center - 16, center - 7), "STOP", fill=(255, 255, 255))

    elif sign_id in [2, 3]:
        # Speed limit sign: circular white sign with red border
        img.paste((180, 195, 205), [0, 0, size, size])
        draw = ImageDraw.Draw(img)
        # Red border circle
        bbox = [center - r, center - r, center + r, center + r]
        draw.ellipse(bbox, fill=(255, 255, 255), outline=(215, 20, 20), width=int(size * 0.1))
        speed_text = "30" if sign_id == 2 else "60"
        draw.text((center - 9, center - 7), speed_text, fill=(20, 20, 20))

    elif sign_id == 4:
        # Yield sign: Inverted triangle
        img.paste((180, 195, 205), [0, 0, size, size])
        draw = ImageDraw.Draw(img)
        pts = [
            (center - r * 1.05, center - r * 0.8),
            (center + r * 1.05, center - r * 0.8),
            (center, center + r * 1.05)
        ]
        draw.polygon(pts, fill=(255, 255, 255), outline=(215, 25, 25), width=int(size * 0.12))
        inner_pts = [
            (center - r * 0.55, center - r * 0.45),
            (center + r * 0.55, center - r * 0.45),
            (center, center + r * 0.6)
        ]
        draw.polygon(inner_pts, fill=(255, 255, 255))

    elif sign_id == 5:
        # Sharp Turn: Yellow diamond with black arrow
        img.paste((180, 195, 205), [0, 0, size, size])
        draw = ImageDraw.Draw(img)
        diamond_r = r * 1.05
        pts = [
            (center, center - diamond_r),
            (center + diamond_r, center),
            (center, center + diamond_r),
            (center - diamond_r, center)
        ]
        draw.polygon(pts, fill=(245, 185, 10), outline=(20, 20, 20), width=2)
        # Arrow curve
        draw.arc([center - 12, center - 8, center + 8, center + 12], start=180, end=270, fill=(20, 20, 20), width=3)
        draw.polygon([(center + 6, center - 13), (center + 12, center - 7), (center + 4, center - 6)], fill=(20, 20, 20))

    if augment:
        # Random slight rotation
        angle = random.uniform(-10, 10)
        img = img.rotate(angle, resample=Image.Resampling.BILINEAR)
        # Random brightness / contrast shift
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(random.uniform(0.7, 1.3))
        # Random subtle blur
        if random.random() > 0.6:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 0.8)))

    return img.resize((size, size))

def render_road_camera_frame(
    curvature: float,
    car_offset: float,
    width: int = CAM_WIDTH,
    height: int = CAM_HEIGHT,
    sign_id: int = 0,
    sign_distance: float = 1.0,
    brightness: float = 1.0,
    add_shadow: bool = False
) -> Image.Image:
    """
    Renders realistic synthetic front-facing dashboard camera view.
    
    curvature: -1.0 (sharp left) to +1.0 (sharp right)
    car_offset: -1.0 (displaced left) to +1.0 (displaced right)
    sign_id: 0 (none) or 1..5 for roadside traffic sign
    sign_distance: 0.1 (far away near horizon) to 1.0 (close to vehicle)
    """
    img = Image.new("RGB", (width, height))
    draw = ImageDraw.Draw(img)

    horizon_y = int(height * 0.42)

    # 1. Sky & Horizon
    for y in range(horizon_y):
        t = y / max(1, horizon_y)
        r = int(SKY_COLOR[0] * (1 - t) + HORIZON_COLOR[0] * t)
        g = int(SKY_COLOR[1] * (1 - t) + HORIZON_COLOR[1] * t)
        b = int(SKY_COLOR[2] * (1 - t) + HORIZON_COLOR[2] * t)
        draw.line([(0, y), (width, y)], fill=(r, g, b))

    # 2. Grass ground
    draw.rectangle([(0, horizon_y), (width, height)], fill=GRASS_COLOR)

    # 3. Perspective Road Surface
    # Bottom road width spans most of frame; top road width narrows near horizon
    bottom_half_width = width * 0.48
    top_half_width = width * 0.08
    
    # Car offset shifts the camera's apparent position relative to lane center
    # Negative offset = car is left -> road appears shifted to right
    center_bottom = (width / 2.0) - (car_offset * width * 0.28)
    
    # Curvature bends the vanishing point at horizon
    center_top = (width / 2.0) + (curvature * width * 0.38) - (car_offset * width * 0.08)

    # Road polygon slices for smooth curved perspective
    num_slices = 16
    for i in range(num_slices):
        t0 = i / num_slices
        t1 = (i + 1) / num_slices
        y0 = int(horizon_y + t0 * (height - horizon_y))
        y1 = int(horizon_y + t1 * (height - horizon_y))

        # Quadratic curve bending
        cx0 = center_top * (1 - t0) + center_bottom * t0 + (curvature * 25.0 * math.sin(t0 * math.pi))
        cx1 = center_top * (1 - t1) + center_bottom * t1 + (curvature * 25.0 * math.sin(t1 * math.pi))

        hw0 = top_half_width * (1 - t0) + bottom_half_width * t0
        hw1 = top_half_width * (1 - t1) + bottom_half_width * t1

        road_poly = [
            (cx0 - hw0, y0),
            (cx0 + hw0, y0),
            (cx1 + hw1, y1),
            (cx1 - hw1, y1)
        ]
        draw.polygon(road_poly, fill=ASPHALT_COLOR)

        # Left edge line (Yellow/White)
        lw0 = max(1, int(1 + t0 * 3))
        draw.line([(cx0 - hw0, y0), (cx1 - hw1, y1)], fill=LANE_YELLOW, width=lw0)
        # Right edge line (White)
        draw.line([(cx0 + hw0, y0), (cx1 + hw1, y1)], fill=LANE_WHITE, width=lw0)

        # Center dashed line
        if (i % 3) != 0:
            draw.line([(cx0, y0), (cx1, y1)], fill=LANE_WHITE, width=max(1, int(t0 * 2)))

    # 4. Roadside Traffic Sign (if sign_id > 0)
    if sign_id > 0:
        # Distance determines sign scaling and y position
        d = max(0.1, min(1.0, sign_distance))
        sy = int(horizon_y + (height - horizon_y) * (d * 0.75))
        # Place on right verge of the road
        t_sign = (sy - horizon_y) / (height - horizon_y)
        cx_sign = center_top * (1 - t_sign) + center_bottom * t_sign + (curvature * 25.0 * math.sin(t_sign * math.pi))
        hw_sign = top_half_width * (1 - t_sign) + bottom_half_width * t_sign
        
        sign_x = int(cx_sign + hw_sign + 8 + d * 14)
        
        # Scale sign based on perspective distance
        sign_pixel_size = max(12, int(8 + d * 24))
        sign_img = generate_traffic_sign(sign_id, size=sign_pixel_size, augment=False)
        
        # Post pole
        pole_bottom = sy
        pole_top = sy - sign_pixel_size
        draw.line([(sign_x + sign_pixel_size // 2, pole_bottom), (sign_x + sign_pixel_size // 2, pole_top)], fill=(120, 120, 120), width=2)
        
        # Paste sign
        box = (sign_x, pole_top - sign_pixel_size // 2)
        if 0 <= box[0] < width and 0 <= box[1] < height:
            img.paste(sign_img, box)

    # 5. Lighting / Shadows / Noise
    if add_shadow:
        # Random shadow diagonal band
        shadow_overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        sdraw = ImageDraw.Draw(shadow_overlay)
        sx = random.randint(0, width // 2)
        sdraw.polygon([(sx, horizon_y), (sx + 40, horizon_y), (sx + 80, height), (sx + 30, height)], fill=(0, 0, 0, 75))
        img = Image.alpha_composite(img.convert("RGBA"), shadow_overlay).convert("RGB")

    if brightness != 1.0:
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(brightness)

    return img
