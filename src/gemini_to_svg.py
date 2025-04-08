from dotenv import load_dotenv
import os, time, csv, json, base64
import pandas as pd
from tqdm import tqdm
from google import genai
from google.genai import types

import pandas as pd

load_dotenv()

# Set your API key - you should replace this with your actual API key or use environment variables
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY', '')  # Get API key from environment variable
if not GOOGLE_API_KEY:
    GOOGLE_API_KEY = input("Please enter your Google Gemini API key: ")
else:
    print("Using API key from environment variable.")


# System prompt for generating SVG images
SYSTEM_PROMPT = """
You are an expert SVG generator. Your task is to analyze a scene description, plan the composition, and generate an SVG using only **allowed tags and attributes**, with **short comments** explaining each visual block.
---
### ⚙️ Constraints
- **Allowed Tags:**  
  `svg`, `path`, `circle`, `rect`, `ellipse`, `line`, `polyline`, `polygon`, `g`, `linearGradient`, `radialGradient`, `stop`, `defs`
- **Allowed Attributes:**  
  `viewBox`, `width`, `height`, `fill`, `stroke`, `stroke-width`, `d`, `cx`, `cy`, `r`, `x`, `y`, `rx`, `ry`, `x1`, `y1`, `x2`, `y2`, `points`, `transform`, `opacity`
- **Output Format:**  
  Always begin the output with:  
  ```xml
  <svg viewBox="0 0 256 256" width="256" height="256" xmlns="http://www.w3.org/2000/svg">
  ```  
  Always close with:  
  ```xml
  </svg>
  ```
- **No `<text>` elements**, and no tags or attributes outside the allowed lists.
- **Add short HTML-style comments (`<!-- ... -->`)** before each visual block explaining what it represents and the color used.
---
### 🧠 Generation Process
Always start with a short reasoning section:
1. Identify visual elements from the prompt  
2. Choose a consistent and expressive color palette  
3. Lay out the composition within the 256×256 viewBox  
4. Create base shapes and background  
5. Add details, layering, and visual interest  
6. Use `<g>` and `transform` for grouping or transformations  
7. Review for clarity, balance, completeness, and constraint compliance  
---
### 🧪 Example
#### Prompt
```
A lighthouse overlooking the ocean.
```
#### Output
```
<reasoning>
1. Key Elements: sky, ocean, lighthouse, light beam, and subtle rocky base.
2. Colors & Gradients: Use smooth gradients for sky (light blue to blue) and ocean (blue to deep blue). The lighthouse features contrasting white and red with gray details. The light beam is a soft yellow.
3. Layout: The sky occupies the top, the ocean the lower half. The lighthouse stands left of center, with the light beam extending toward the right.
4. Base Shapes: Use <rect> for sky and ocean; <polygon>, <rect>, and <circle> for the lighthouse; <path> for the light beam; and <ellipse> for rocks.
5. Details: Introduce window highlights and gradient backgrounds, ensuring smooth transitions.
6. Transformations: Group related elements using <g> and use gradients via <defs>.
7. Review: The composition is balanced and visually appealing with clear color distinctions and soft, gradient effects.
</reasoning>

<svg viewBox="0 0 256 256" width="256" height="256" xmlns="http://www.w3.org/2000/svg">

  <!-- Define gradients for sky and ocean -->
  <defs>
    <!-- Sky gradient: from pale blue to richer blue -->
    <linearGradient id="skyGradient" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="lightblue" />
      <stop offset="100%" stop-color="deepskyblue" />
    </linearGradient>
    <!-- Ocean gradient: from blue to deep navy -->
    <linearGradient id="oceanGradient" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="blue" />
      <stop offset="100%" stop-color="navy" />
    </linearGradient>
  </defs>

  <!-- Sky background using gradient -->
  <rect x="0" y="0" width="256" height="128" fill="url(#skyGradient)" />

  <!-- Ocean using gradient -->
  <rect x="0" y="128" width="256" height="128" fill="url(#oceanGradient)" />

  <!-- Lighthouse group -->
  <g>
    <!-- Lighthouse body (white) -->
    <rect x="50" y="80" width="30" height="120" fill="white" stroke="gray" stroke-width="1"/>
    <!-- Red stripe -->
    <rect x="50" y="130" width="30" height="20" fill="red" opacity="0.8"/>
    <!-- Lantern room (top of lighthouse) -->
    <circle cx="65" cy="80" r="8" fill="gray" />
  </g>

  <!-- Light beam (soft yellow triangle) -->
  <path d="M65 80 L130 60 L130 100 Z" fill="yellow" opacity="0.3" />
  
  <!-- Rocks at lighthouse base -->
  <g>
    <ellipse cx="60" cy="205" rx="10" ry="4" fill="dimgray" />
    <ellipse cx="75" cy="210" rx="8" ry="3" fill="slategray" />
  </g>

</svg>
```
"""

def generate_svg(description, model_name = "gemini-2.0-flash"):
    """
    Generate SVG content using Gemini model based on a text description
    """
    try:
        # Configure the model
        # model = genai.GenerativeModel(model_name)

        client = genai.Client(
            api_key=os.environ.get("GEMINI_API_KEY"),
        )

        model = "gemini-2.0-flash"
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=f"Create an SVG image based on this description: {description}"),
                ],
            ),
        ]
        generate_content_config = types.GenerateContentConfig(
            response_mime_type="text/plain",
            system_instruction=[
                types.Part.from_text(text=SYSTEM_PROMPT),
            ],
        )

        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=generate_content_config,
        )
        
        # Extract SVG content from response
        svg_content = response.text
        
        # Basic validation to ensure the response contains SVG
        if not "</svg>" in svg_content:
            return f"Error: Invalid SVG generated: {svg_content[:100]}..."
        
        return svg_content
    
    except Exception as e:
        return f"Error: {str(e)}"


def process_csv(input_csv_path, output_path):
    """
    Process the input CSV file, generate SVGs for each description,
    and save the results to the output JSONL file
    """
    # Read the input CSV file
    try:
        df = pd.read_csv(input_csv_path)
    except Exception as e:
        print(f"Error reading input file: {e}")
        return
    
    # Ensure output path has .jsonl extension
    if not output_path.lower().endswith('.jsonl'):
        output_path = output_path.rsplit('.', 1)[0] + '.jsonl'
        print(f"Changed output format to JSONL: {output_path}")
    
    # Create empty output file (or truncate existing file)
    with open(output_path, 'w', encoding='utf-8') as f:
        pass
    
    # Process each row with a tqdm progress bar
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Generating SVGs"):
        id_value = row['id']
        description = row['description']
        
        # Generate SVG for current description
        svg_content = generate_svg(description)
        
        # Create JSON object
        json_obj = {
            'id': id_value,
            'description': description,
            'svg_content': svg_content
        }
        
        # Open file in append mode, write, and close
        with open(output_path, 'a', encoding='utf-8') as f_output:
            f_output.write(json.dumps(json_obj) + '\n')
        
        # Add a small delay to avoid hitting rate limits
        time.sleep(0.5)
    
    print(f"Processing complete. Results saved to {output_path}")

    ### doing some additional cleanup
    print("Cleaning up the SVG content...")
    try :
        # Read the output JSONL file
        df = pd.read_json(output_path, lines=True)
        # remove the ```xml and ``` in the svg_content column
        df['svg_content'] = df['svg_content'].str.replace('```xml', '', regex=False).str.replace('```', '', regex=False)
        df['svg_content'] = df['svg_content'].str.strip()
        # save it back to a new jsonl file
        df.to_json(output_path, orient='records', lines=True)

    except Exception as e:
        print(f"Error cleaning up SVG content: {e}")
        return
    
    print(f"Saved cleaned data to {output_path}")

if __name__ == "__main__":
    # File paths
    input_csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                "data", "llm_prompt.csv")
    output_csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                 "data", "svg_results.csv")
    
    # Process the CSV file
    process_csv(input_csv_path, output_csv_path)
