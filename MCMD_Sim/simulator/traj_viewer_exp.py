import dash # type: ignore
from dash import Dash, dcc, html, Input, Output, State # type: ignore
import plotly.graph_objects as go # type: ignore
import numpy as np
import pandas as pd
import json
import os
from .components.simobjects import SimObject
from .utils.logger import SimLogger

app = Dash(__name__)

def get_style_plotly(obj_type: str, color: str) -> str:
    """Return matplotlib style string based on object type and color."""
    # Default color and shape fallbacks
    color_map = {
        "red": "r",
        "blue": "b",
        "yellow": "y",
        "green": "g",
        "purple": "m",  # using magenta for purple
    }

    marker_map = {
        "ball": "o",
        "cube": "s",
        "pyramid": "^",
        "jeep": "P",
        "horse": "*",
        "dog": "X",
        "palmtree": "v",
        "watermelon": "D",
        "rocket": ">"
    }

    c = color_map.get(color, "k")
    m = marker_map.get(obj_type, ".")
    return c + m

mpl_to_plotly_colors = {
    'r': 'red', 'b': 'blue', 'y': 'yellow', 'g': 'green', 'm': 'purple', 'k': 'black'
}

mpl_to_plotly_symbols = {
    'o': 'circle',
    's': 'square',
    '^': 'diamond-open',
    'P': 'cross',
    '*': 'square-open',
    'X': 'x',
    'v': 'diamond-open',
    'D': 'diamond',
    '>': 'circle-open',
    '.': 'circle-open'
}

def load_scene(folder):
    init_cond = json.load(open(os.path.join(folder, 'init_conditions.json'), 'r'))
    pos_arr = pd.read_csv(os.path.join(folder, 'data', 'pos.csv')).to_numpy()
    pos_data = pos_arr[:, :3]
    yaw_data = pos_arr[:, 3]
    objs = [SimObject(lrel, init_cond['theta_environment'], colr, otype)
        for lrel, colr, otype in zip(init_cond['objects_loc'], init_cond['objects_color'], init_cond['objects_type'])]
    objs = [SimLogger.parse_obj(obj) for obj in objs]
    return pos_data, yaw_data, objs, init_cond['theta_environment']

def render_scene(pos_data, yaw_data, theta_env, objs=None):
    fig = go.Figure()

    # 3D trajectory line
    fig.add_trace(go.Scatter3d(
        x=pos_data[:, 0], y=pos_data[:, 1], z=pos_data[:, 2],
        mode='lines+markers',
        line=dict(color='blue'),
        marker=dict(size=3),
        name='Trajectory'
    ))

    # Add yaw arrows as simple lines
    step = 5
    arrow_len = 0.2  # length of direction line

    for i in range(0, len(yaw_data), step):
        x, y, z = pos_data[i]
        ang = yaw_data[i] - theta_env + 2 * np.pi
        dx, dy = arrow_len * np.cos(ang), arrow_len * np.sin(ang)

        fig.add_trace(go.Scatter3d(
            x=[x, x + dx],
            y=[y, y + dy],
            z=[z, z],  # flat in XY plane
            mode='lines',
            line=dict(color='red', width=4),
            showlegend=False
        ))
        fig.add_trace(go.Scatter3d(
            x=[x + dx],
            y=[y + dy],
            z=[z],
            mode='markers',
            marker=dict(color='red', size=3),
            showlegend=False
        ))

    # # Add yaw arrows (as cones in 3D but flat on xy-plane)
    # step = 5
    # for i in range(0, len(yaw_data), step):
    #     x, y, z = pos_data[i]
    #     ang = yaw_data[i] - theta_env + 2 * np.pi
    #     dx, dy = 0.2 * np.cos(ang), 0.2 * np.sin(ang)
    #     fig.add_trace(go.Cone(
    #         x=[x], y=[y], z=[z],
    #         u=[dx], v=[dy], w=[0],  # w=0 → flat on XY
    #         sizemode="absolute",
    #         sizeref=0.1,
    #         anchor="tail",
    #         colorscale="Reds",
    #         showscale=False,
    #         name="Yaw Arrows"
    #     ))

    # Plot optional objects
    if objs is not None:
        for obj in objs:
            ox, oy, oz = obj['x'], obj['y'], obj.get('z', 0)
            # Extract style
            style = obj.get('style', 'ro')  # fallback to red circle
            color_code = style[0]
            symbol_code = style[1] if len(style) > 1 else 'o'

            color = mpl_to_plotly_colors.get(color_code, 'black')
            symbol = mpl_to_plotly_symbols.get(symbol_code, 'circle')
            fig.add_trace(go.Scatter3d(
                x=[ox], y=[oy], z=[oz],
                mode='markers',
                marker=dict(size=5, color=color, symbol=symbol),
                name=obj.get('label', 'Object')
            ))
            # Add a transparent sphere/circle (not a perfect circle, but an approximation)
            u = np.linspace(0, 2 * np.pi, 30)
            cx = ox + 0.5 * np.cos(u)
            cy = oy + 0.5 * np.sin(u)
            cz = np.ones_like(u) * oz
            fig.add_trace(go.Scatter3d(
                x=cx, y=cy, z=cz,
                mode='lines',
                line=dict(color='red', dash='dot'),
                showlegend=False
            ))

    fig.update_layout(
        scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z', 
                   aspectmode='data',
                   camera=dict(
                        eye=dict(x=0.0724, y=2.11, z=1.13),  # <- main view tweak here
                        up=dict(x=0, y=0, z=1.0),
                        projection=dict(type='perspective'),
                        center=dict(x=0, y=0, z=0)
                    )),
        title='3D Trajectory Viewer',
        margin=dict(l=0, r=0, b=0, t=40),
        showlegend=True
    )
    return fig

# Layout
app.layout = html.Div([
    html.Div([
        dcc.Input(id='folder-input', type='text', placeholder='Enter base folder path, --(train/eval) for collected dataset',
                style={'width': '60%', 'marginRight': '10px'}),
        html.Button("Load Scenes", id='load-btn', n_clicks=0),
    ], style={'marginBottom': '10px'}),

    html.Div([
        dcc.Input(id='filter-input', type='text', placeholder='Optional filters (comma-separated)',
                style={'width': '60%'})
    ], style={'marginBottom': '20px'}),

    html.Div(id="scene-label", style={"margin": "10px", "fontSize": "20px"}),
    dcc.Graph(id='scene-viewer'),

    html.Div([
        html.Button("Prev", id='prev-btn', n_clicks=0),
        html.Button("Next", id='next-btn', n_clicks=0)
    ]),

    dcc.Store(id='scene-index', data=0),
    dcc.Store(id='subfolders', data=[]),
    dcc.Store(id='base-path', data=None)
])

# Load button callback
@app.callback(
    Output('subfolders', 'data'),
    Output('base-path', 'data'),
    Output('scene-index', 'data', allow_duplicate=True),       # NEW: also reset index to 0
    Output('scene-viewer', 'figure', allow_duplicate=True),    # NEW: plot first scene right away
    Output('scene-label', 'children', allow_duplicate=True),   # NEW: update label right away
    Input('load-btn', 'n_clicks'),
    State('folder-input', 'value'),
    State('filter-input', 'value'),
    prevent_initial_call=True
)
def load_folder_list(n_clicks, folder_path, filters):
    if not folder_path or not os.path.isdir(folder_path):
        return [], None, 0, go.Figure(), "Invalid path"

    subfolders = sorted([
        f for f in os.listdir(folder_path)
        if os.path.isdir(os.path.join(folder_path, f))
    ])
    if filters:
        filters = [f.strip() for f in filters.split(',')]
        subfolders = [
            pfldr for pfldr in subfolders
            if all(any(fld_detail.startswith(f) or fld_detail.endswith(f) for fld_detail in pfldr.split('_'))
                   for f in filters)
        ]

    if not subfolders:
        return [], folder_path, 0, go.Figure(), "No matching subfolders"

    folder = os.path.join(folder_path, subfolders[0])
    pos, yaw, objs, theta_env = load_scene(folder)
    fig = render_scene(pos, yaw, theta_env, objs)

    return subfolders, folder_path, 0, fig, f"Scene: {subfolders[0]}"



# Scene rendering callback
@app.callback(
    Output('scene-viewer', 'figure'),
    Output('scene-label', 'children'),
    Output('scene-index', 'data', allow_duplicate=True),
    Input('next-btn', 'n_clicks'),
    Input('prev-btn', 'n_clicks'),
    State('scene-index', 'data'),
    State('subfolders', 'data'),
    State('base-path', 'data'),
    prevent_initial_call=True
)
def update_scene(next_clicks, prev_clicks, current_index, subfolders, base_path):
    if not subfolders or not base_path:
        return go.Figure(), "No scene loaded", 0

    triggered = dash.callback_context.triggered_id
    if triggered == 'next-btn':
        current_index = (current_index + 1) % len(subfolders)
    elif triggered == 'prev-btn':
        current_index = (current_index - 1) % len(subfolders)

    folder = os.path.join(base_path, subfolders[current_index])
    pos, yaw, objs, theta_env = load_scene(folder)
    fig = render_scene(pos, yaw, theta_env, objs)
    return fig, f"Scene: {subfolders[current_index]}", current_index

if __name__ == '__main__':
    app.run(debug=True)
