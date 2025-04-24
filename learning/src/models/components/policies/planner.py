from openai import OpenAI

client = OpenAI(api_key="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")
import json

# Set your API key securely

# Load the base prompt from file or as a long string
BASE_PROMPT = """
                    You are a robotics planning assistant that extracts structured navigation steps from natural language instructions. You always reply in **English**, regardless of the input language.

                    Your task is to convert the input into a precise, logical sequence of robot navigation steps. Each step should be grounded in physically plausible, vision-detectable targets (like “log”, “shed”, “tree”, “path”).

                    You must output a **valid JSON array**, where each item contains:
                    - `"step"`: the step number, starting at 1
                    - `"action"`: one of ["Toward", "Above", "Below", "Left", "Right", "Turn"]
                    - `"target"`: a short, descriptive phrase naming a visual object or feature

                    ---

                    **Turning Instructions**
                    - If the user says to turn, use `"action": "Turn"` and describe the direction and angle in `"target"` (e.g., `"Left 90°"`)
                    - If no angle is specified, assume 90°

                    **Rules**
                    - Always output valid JSON — no comments, no explanations
                    - Always reply in **English**, even if the input is in another language
                    - Use spatial reasoning and common sense to break down informal or vague phrasing
                    - Never use vague targets like "somewhere", "continue", or "keep going"
                    - If uncertain, provide the most reasonable grounded interpretation

                    ⚠️ Your output **must be only** a valid JSON list — do not include preambles, titles, or comments.

                    ---

                    ### Multilingual Input Examples

                    Input:
                    1. **English** — "Fly through the gap between the two trees, then above a large fallen log. Turn left at the dirt path, then head toward the small wooden shed."
                    2. **German** — "Flieg durch die Lücke zwischen den zwei Bäumen und fliege dann über einen großen umgefallenen Baumstamm. Biege am Trampelpfad links ab und fliege dann auf den kleinen Holzschuppen zu."
                    3. **Spanish** — "Vuela por la abertura entre los dos árboles, luego pasa por encima de un tronco grande caído. Gira a la izquierda en el sendero de tierra y dirígete hacia el pequeño cobertizo de madera."
                    4. **Arabic** — "طر بين الشجرتين ثم فوق جذع شجرة كبير سقط على الأرض. انعطف يسارًا عند الممر الترابي، ثم اتجه نحو الكوخ الخشبي الصغير."

                    Output:
                    [
                    { "step": 1, "action": "Toward", "target": "Tree gap" },
                    { "step": 2, "action": "Above", "target": "Large fallen log" },
                    { "step": 3, "action": "Turn", "target": "Left 90°" },
                    { "step": 4, "action": "Toward", "target": "Dirt path" },
                    { "step": 5, "action": "Toward", "target": "Small wooden shed" }
                    ]

                    ---

                    Now process the following instruction:

                    {instruction}
                    """

def build_prompt(instruction: str) -> str:
    """Insert the user-provided instruction into the base prompt."""
    return BASE_PROMPT.replace("{instruction}", instruction.strip())

def call_planner(instruction: str, model="gpt-4", temperature=0.0):
    """Call GPT-4 API with navigation instruction and return parsed JSON."""
    prompt = build_prompt(instruction)

    response = client.chat.completions.create(model=model,
    temperature=temperature,
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ])

    reply = response.choices[0].message.content.strip()

    try:
        plan = json.loads(reply)
        if not isinstance(plan, list):
            raise ValueError("Expected a list of steps.")
        return plan
    except json.JSONDecodeError as e:
        print("⚠️ Failed to parse JSON output:")
        print(reply)
        raise e

# Example usage
if __name__ == "__main__":
    from fuzzywuzzy import fuzz

    # Expected step structure
    expected_steps = [
        { "step": 1, "action": "Toward", "target": "Red flag" },
        { "step": 2, "action": "Turn", "target": "Right 90°" },
        { "step": 3, "action": "Toward", "target": "Large tree" },
        { "step": 4, "action": "Turn", "target": "Left 90°" },
        { "step": 5, "action": "Above", "target": "Picnic table" },
        { "step": 6, "action": "Toward", "target": "Yellow barrel" },
        { "step": 7, "action": "Below", "target": "Rope bridge" },
        { "step": 8, "action": "Above", "target": "Checkered mat" }
    ]


    instructions = [
        # English - canonical
        "Fly toward the red flag. Turn right. Head toward the large tree. Turn left. Fly above the picnic table. Move toward the yellow barrel. Fly below the rope bridge. Hover above the checkered mat.",

        # English - varied
        "Go to the red flag, then make a right turn. Proceed to the big tree and take a left. Pass over the picnic table. Head to the yellow barrel. Fly under the rope bridge and stop over the checkered mat.",

        # Spanish
        "Dirígete hacia la bandera roja. Gira a la derecha. Luego hacia el árbol grande. Gira a la izquierda. Vuela por encima de la mesa de picnic. Avanza hacia el barril amarillo. Pasa por debajo del puente de cuerda. Quédate flotando sobre la alfombra a cuadros.",

        # German
        "Flieg zum roten Fahne. Dann rechts abbiegen. Flieg zum großen Baum. Dann links abbiegen. Flieg über den Picknicktisch. Dann zum gelben Fass. Flieg unter der Seilbrücke hindurch. Schwebe über der karierten Matte.",

        # Arabic
        "اتجه نحو العلم الأحمر. انعطف يمينًا. اتجه نحو الشجرة الكبيرة. انعطف يسارًا. حلق فوق طاولة النزهة. اتجه نحو البرميل الأصفر. مر من تحت جسر الحبال. توقف فوق الحصيرة المربعة.",

        # English - minimal
        "Red flag. Right. Large tree. Left. Above picnic table. Yellow barrel. Below rope bridge. Hover over mat.",

        # English - slightly ambiguous
        "Fly to the red flag. Turn right. Keep going to the tree. Left turn. Over the picnic table. Toward the yellow barrel. Under the bridge. Stop above the checkered mat.",

        # English - broken down
        "1. Go to red flag. 2. Right turn. 3. To large tree. 4. Left turn. 5. Above picnic table. 6. To yellow barrel. 7. Below rope bridge. 8. Hover above mat.",

        # Spanish - casual phrasing
        "Ve hacia la bandera roja. Gira a la derecha. Sigue hasta el árbol grande. Gira a la izquierda. Pasa por encima de la mesa de picnic. Luego al barril amarillo. Pasa por debajo del puente. Quédate sobre la alfombra.",

        # English - descriptive
        "Start by heading to the red flag. Turn 90 degrees to your right. Then approach the large tree and make a left. Fly above the picnic table. Head straight to the yellow barrel. Pass under the rope bridge. Hover directly above the checkered mat."
    ]


    FUZZY_THRESHOLD = 55

    def fuzzy_match(s1, s2, threshold=FUZZY_THRESHOLD):
        return fuzz.ratio(s1.lower(), s2.lower()) >= threshold

    def test_fuzzy(instructions, expected_steps):
        for i, instr in enumerate(instructions, 1):
            print(f"\n=== Test Case {i} ===")
            try:
                steps = call_planner(instr)

                if len(steps) != len(expected_steps):
                    print(f"❌ Step count mismatch: expected {len(expected_steps)}, got {len(steps)}")
                    print(json.dumps(steps, indent=2))
                    continue

                match_fail = False
                for j, (step, expected) in enumerate(zip(steps, expected_steps), 1):
                    act_ok = step["action"] == expected["action"]
                    tgt_ok = fuzzy_match(step["target"], expected["target"])

                    if not act_ok or not tgt_ok:
                        print(f"❌ Step {j} mismatch:")
                        print(f"  Action  → Got '{step['action']}', expected '{expected['action']}'")
                        print(f"  Target  → Got '{step['target']}', expected '{expected['target']}'")
                        # print the fuzzy match score
                        print(f"  Fuzzy match score: {fuzz.ratio(step['target'].lower(), expected['target'].lower())}")
                        match_fail = True

                if not match_fail:
                    print("✅ Actions correct, targets fuzzy-matched")
            except Exception as e:
                print(f"❌ Exception: {e}")

    test_fuzzy(instructions, expected_steps)

