import os
import time
import sys
import shutil
import json
import mimetypes
from pathlib import Path
from typing import Optional, List
from google import genai
from google.genai import types as gtypes

# Try to import PIL for the deterministic grid generation
try:
    from PIL import Image, ImageDraw
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("⚠️ PIL (Pillow) not found. Install it for perfect board alignment: 'pip install pillow'")

# --- CONFIGURATION ---
API_KEY = os.environ.get("GEMINI_API_KEY")

# ============================================================================
# 1) BANANA (Image Gen)
# ============================================================================
def banana_generate(prompt: str, input_paths: Optional[List[str]] = None, 
                    out_dir: str = ".", n: int = 1,
                    model: str = "gemini-3-pro-image-preview"):
    
    if not API_KEY:
        print("❌ GEMINI_API_KEY not found.")
        sys.exit(1)

    client = genai.Client(api_key=API_KEY)
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    parts = [gtypes.Part.from_text(text=prompt)]
    input_paths = input_paths or []

    for p in input_paths:
        if os.path.exists(p):
            with open(p, "rb") as f:
                data = f.read()
            mt, _ = mimetypes.guess_type(p)
            parts.append(gtypes.Part.from_bytes(data=data, mime_type=mt or "image/png"))

    contents = [gtypes.Content(role="user", parts=parts)]
    config = gtypes.GenerateContentConfig(response_modalities=["IMAGE", "TEXT"])

    out_paths = []
    try:
        stream = client.models.generate_content_stream(
            model=model, contents=contents, config=config
        )

        idx = 0
        for chunk in stream:
            cand = getattr(chunk, "candidates", None)
            if not cand or not cand[0].content or not cand[0].content.parts:
                continue

            p = cand[0].content.parts[0]
            inline = getattr(p, "inline_data", None)
            
            if inline and inline.data:
                fname = f"banana_{int(time.time()*1000)}_{idx}.png"
                fpath = os.path.join(out_dir, fname)
                with open(fpath, "wb") as f:
                    f.write(inline.data)
                out_paths.append(fpath)
                idx += 1
                if len(out_paths) >= n:
                    break
    except Exception as e:
        # Graceful Quota Handling
        if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
            print("\n🚨 BANANA QUOTA EXCEEDED (429).")
            print("   The script is stopping safely.")
            print("   ✅ Run this script again later/tomorrow to resume exactly here.")
            sys.exit(0)
        print(f"❌ Generation Error: {e}")
        
    return out_paths

# ============================================================================
# 2) VEO (Video Gen)
# ============================================================================
def veo_generate_video(prompt: str, image_path: Optional[str], out_dir: str = ".",
                       aspect_ratio="16:9", resolution="720p",
                       model="veo-3.1-generate-preview"):
    
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    client = genai.Client(api_key=API_KEY)

    image_obj = None
    if image_path and os.path.exists(image_path):
        with open(image_path, "rb") as f:
            data = f.read()
        mt, _ = mimetypes.guess_type(image_path)
        image_obj = gtypes.Image(image_bytes=data, mime_type=mt or "image/png")

    cfg = gtypes.GenerateVideosConfig(
        aspect_ratio=aspect_ratio, resolution=resolution
    )

    print(f"   ...sending to Veo...")
    try:
        op = client.models.generate_videos(
            model=model, prompt=prompt, image=image_obj, config=cfg
        )
    except Exception as e:
        # Graceful Quota Handling
        if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
            print("\n🚨 VEO QUOTA EXCEEDED (429).")
            print("   The script is stopping safely.")
            print("   ✅ Run this script again later/tomorrow to resume exactly here.")
            sys.exit(0)
        print(f"❌ Veo Start Error: {e}")
        return None

    # Poll
    while not op.done:
        time.sleep(6)
        try:
            op = client.operations.get(op)
        except:
            pass

    if not op.result or not op.result.generated_videos:
        return None

    video = op.result.generated_videos[0].video
    out_path = os.path.join(out_dir, f"veo_{int(time.time()*1000)}.mp4")
    
    try:
        video_bytes = client.files.download(file=video)
        with open(out_path, "wb") as f:
            f.write(video_bytes)
    except Exception as e:
        print(f"❌ Save Error: {e}")
        return None
        
    return out_path

# ============================================================================
# 3) PIPELINE STEPS
# ============================================================================

def create_guide_board(out_path, size=1024):
    """Creates a perfect black and white checkerboard to guide the AI."""
    if not PIL_AVAILABLE: return None
    
    print(f"   📐 Creating deterministic guide grid...")
    img = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(img)
    square_size = size // 8
    
    for r in range(8):
        for c in range(8):
            if (r + c) % 2 == 1:
                x = c * square_size
                y = r * square_size
                draw.rectangle([x, y, x + square_size, y + square_size], fill="black")
    
    img.save(out_path)
    return out_path

def generate_board_texture(base_dir, board_desc, temp_dir):
    """Generates the single main board texture using a guide grid."""
    
    # --- RESUME LOGIC START ---
    final_path = os.path.join(base_dir, "board_texture.png")
    if os.path.exists(final_path):
        print(f"   Skipping Board Texture (Exists): {final_path}")
        return final_path
    # --- RESUME LOGIC END ---

    print(f"\n🎨 Generating Board Texture for '{board_desc}'...")
    
    guide_path = os.path.join(temp_dir, "guide_grid.png")
    created_guide = create_guide_board(guide_path)
    
    prompt = (
        f"Top-down view of a full chess board. Material: {board_desc}. "
        f"Strictly follow the checkerboard pattern of the reference image. "
        f"Perfect 8x8 grid, flat texture, no perspective distortion, orthographic view."
    )
    
    inputs = [created_guide] if created_guide else []
    paths = banana_generate(prompt, input_paths=inputs, out_dir=base_dir, n=1)
    
    if paths:
        if os.path.exists(final_path): os.remove(final_path)
        os.rename(paths[0], final_path)
        print(f"   ✅ Saved {final_path}")
        return final_path
    
    print("   ❌ Failed to generate board.")
    return None

def generate_piece_sprites(base_dir, pieces_dir, temp_dir, piece_desc, board_desc, board_path):
    print(f"\n♟️  Generating Piece Sprites: '{piece_desc}'...")
    Path(pieces_dir).mkdir(parents=True, exist_ok=True)

    pieces = ["pawn", "rook", "knight", "bishop", "queen", "king"] 
    colors = ["white", "black"]

    for color in colors:
        for piece in pieces:
            filename = f"{color}_{piece}.png"
            filepath = os.path.join(pieces_dir, filename)
            
            if os.path.exists(filepath):
                print(f"   Skipping {filename} (exists)")
                continue

            print(f"   Generating {color} {piece}...")
            
            if color == "white":
                bg_prompt = "Isolated on a pure solid BLACK background."
            else:
                bg_prompt = "Isolated on a pure solid WHITE background."

            prompt = (
                f"Studio product photography of a single {color} chess {piece} piece. "
                f"Style: {piece_desc}. "
                f"Use the provided board image (Image 1) as a material reference only. "
                f"{bg_prompt} "
                f"Full body shot, centered, symmetrical, front view. "
                f"NO smoke, NO dust, NO debris, NO magical effects in this shot. "
                f"Clean, sharp edges for cutout."
            )
            
            inputs = [board_path] if board_path else []
            paths = banana_generate(prompt, input_paths=inputs, out_dir=temp_dir, n=1)
            
            if paths:
                if os.path.exists(filepath): os.remove(filepath)
                os.rename(paths[0], filepath)
                print(f"   ✅ Saved {filename}")
            else:
                print(f"   ❌ Failed {filename}")

def generate_animations(base_dir, video_dir, pieces_dir, temp_dir, board_path, anim_desc, piece_desc):
    """
    Generates ALL animations, including King Deaths.
    ENFORCED LOGIC: Attacker is always LEFT, Victim is always RIGHT.
    """
    print(f"\n⚔️  Generating Kill Animations: '{anim_desc}'...")
    
    pieces = ["pawn", "rook", "knight", "bishop", "queen", "king"]
    colors = [("white", "black"), ("black", "white")]

    for a_color, v_color in colors:
        for attacker in pieces:
            for victim in pieces:
                if attacker == "king" and victim == "king": continue

                fname = f"{a_color}_{attacker}_takes_{v_color}_{victim}.mp4"
                out = os.path.join(video_dir, fname)
                
                if os.path.exists(out):
                    print(f"   Skipping {fname} (Exists)")
                    continue

                print(f"   Processing: {a_color} {attacker} vs {v_color} {victim}")
                
                attacker_path = os.path.join(pieces_dir, f"{a_color}_{attacker}.png")
                victim_path = os.path.join(pieces_dir, f"{v_color}_{victim}.png")
                
                inputs = []
                if board_path: inputs.append(board_path)
                if os.path.exists(attacker_path): inputs.append(attacker_path)
                if os.path.exists(victim_path): inputs.append(victim_path)

                # Step 1: Setup Shot (Strict Spatial composition)
                setup_prompt = (
                    f"Cinematic side profile battle shot. "
                    f"On the LEFT side: The {a_color} {attacker} (Aggressor). "
                    f"On the RIGHT side: The {v_color} {victim} (Defender). "
                    f"They are facing each other on the chessboard. "
                    f"Style: {piece_desc}. "
                    f"The {a_color} piece on the left looks powerful and ready to strike. "
                    f"The {v_color} piece on the right looks vulnerable."
                )
                
                setup_png = banana_generate(setup_prompt, input_paths=inputs, out_dir=temp_dir, n=1)
                
                if setup_png:
                    # Step 2: Animation
                    if victim == "king":
                        action = f"The {a_color} {attacker} (LEFT) delivers a fatal blow. The {v_color} King (RIGHT) falls and dies."
                    else:
                        action = f"The {a_color} {attacker} (LEFT) strikes and completely destroys the {v_color} {victim} (RIGHT)."

                    kill_prompt = (
                        f"{action} "
                        f"Action style: {anim_desc}. "
                        f"The piece on the LEFT MUST WIN. The piece on the RIGHT MUST LOSE. "
                        f"Start from the exact visual composition of the provided image. "
                        f"Violent, cinematic physics destruction of the victim only."
                    )
                    out_vid = veo_generate_video(kill_prompt, image_path=setup_png[0], out_dir=temp_dir)
                    if out_vid:
                        os.rename(out_vid, out)
                        print(f"   🎥 Saved {fname}")
                        
                        # IMPORTANT: Quota protection sleep
                        print("   (Cooling down for 15s to respect API limits...)")
                        time.sleep(15) 

def update_theme_manifest(theme_name):
    manifest_path = os.path.join("assets", "themes.json")
    themes = []
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, "r") as f:
                themes = json.load(f)
        except: themes = []
    if theme_name not in themes:
        themes.append(theme_name)
        with open(manifest_path, "w") as f:
            json.dump(themes, f)
        print(f"📝 Registered theme '{theme_name}' in assets/themes.json")

def main():
    print("=========================================")
    print("   BATTLE CHESS ASSET GENERATOR v3.3   ")
    print("   *** QUOTA-SAFE PRODUCTION MODE *** ")
    print("=========================================")
    
    default_theme = "obsidian_gothic"
    theme_title = input(f"\nTheme Title (folder name) [default: {default_theme}]: ").strip() or default_theme
    theme_title = "".join([c for c in theme_title if c.isalnum() or c in ('_', '-')]).lower()
    
    # 1. UPDATE MANIFEST IMMEDIATELY
    # This ensures the theme shows up in the frontend even if the script crashes later.
    base_dir = os.path.join("assets", theme_title)
    pieces_dir = os.path.join(base_dir, "pieces")
    video_dir = os.path.join(base_dir, "videos")
    temp_dir = os.path.join(base_dir, "temp")
    
    for d in [base_dir, pieces_dir, video_dir, temp_dir]:
        Path(d).mkdir(parents=True, exist_ok=True)
    
    print(f"\n📂 Asset Directory: {base_dir}")
    update_theme_manifest(theme_title)

    # --- PROMPT MANAGEMENT ---
    prompts_file = os.path.join(base_dir, "prompts.json")
    saved_prompts = {}
    if os.path.exists(prompts_file):
        try:
            with open(prompts_file, "r") as f:
                saved_prompts = json.load(f)
            print(f"   ℹ️ Loaded saved prompts from {prompts_file}")
        except: pass

    default_board = saved_prompts.get("board_desc", "obsidian and white marble, ancient runes, glowing cracks")
    board_desc = input(f"Board Style [default: {default_board[:20]}...]: ").strip() or default_board
    
    default_piece = saved_prompts.get("piece_desc", "animated stone statues, glowing eyes, dark fantasy armor")
    piece_desc = input(f"Piece Style [default: {default_piece[:20]}...]: ").strip() or default_piece
    
    default_anim = saved_prompts.get("anim_desc", "violent shattering, magic explosions, debris flying, screen shake")
    anim_desc = input(f"Animation Style [default: {default_anim[:20]}...]: ").strip() or default_anim

    # Save current prompts
    try:
        with open(prompts_file, "w") as f:
            json.dump({
                "board_desc": board_desc,
                "piece_desc": piece_desc,
                "anim_desc": anim_desc
            }, f, indent=2)
    except Exception as e:
        print(f"   ⚠️ Could not save prompts: {e}")

    # 2. Execution
    board_png = generate_board_texture(base_dir, board_desc, temp_dir)
    
    if board_png:
        generate_piece_sprites(base_dir, pieces_dir, temp_dir, piece_desc, board_desc, board_path=board_png)
        generate_animations(base_dir, video_dir, pieces_dir, temp_dir, board_path=board_png, anim_desc=anim_desc, piece_desc=piece_desc)
    else:
        print("❌ Board generation failed. Exiting.")

    print("\n✅ GENERATION COMPLETE!")
    print(f"   Run 'python -m http.server' and open the updated game.html")

if __name__ == "__main__":
    main()