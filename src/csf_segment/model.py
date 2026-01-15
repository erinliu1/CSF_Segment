import torch
import torch.nn as nn

class MyFusion(nn.Module):

    def __init__(
        self,
        lstm_hidden_size=64,
        lstm_layers=2,
        lstm_dropout=0.2,
        mlp_features=None,
        mlp_hidden_dims=[64, 32],
        fusion_hidden_dim=64,
    ):
        super().__init__()

        self.lstm_hidden_size = lstm_hidden_size

        # -------- Temporal encoder --------
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=lstm_dropout
        )

        # -------- Static encoder --------
        self.mlp = MyMLP(
            mlp_features=mlp_features,
            hidden_dims=mlp_hidden_dims,
            output_sigmoid=False
        )

        mlp_output_dim = mlp_hidden_dims[-1]

        # -------- FiLM generators --------
        self.film_gamma = nn.Linear(mlp_output_dim, lstm_hidden_size)
        self.film_beta  = nn.Linear(mlp_output_dim, lstm_hidden_size)

        # -------- Final classifier --------
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden_size, fusion_hidden_dim),
            nn.ReLU(),
            nn.Linear(fusion_hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, X_ts, X_static):
        """
        X_ts:     (B, T, 1)
        X_static: (B, static_dim)
        """

        # ----- Time-series path -----
        X_ts = X_ts.unsqueeze(-1)               # (B, T, 1)
        lstm_out, _ = self.lstm(X_ts)   # (B, T, H) 

        # ----- Static path -----
        context = self.mlp(X_static)           # (B, mlp_dim)

        # ----- FiLM conditioning -----
        gamma = self.film_gamma(context).unsqueeze(1)       # (B, 1, H)
        beta  = self.film_beta(context).unsqueeze(1)        # (B, 1, H)

        modulated = gamma * lstm_out + beta     # (B, T, H)

        pooled = torch.mean(modulated, dim=1)  # (B, H)

        # ----- Final prediction -----
        prob = self.classifier(pooled)
        
        return prob.view(-1)

class MyMLP(nn.Module):
    """
    Flexible MLP for static voxel features.

    Features can include any subset of:
        - 'sbref'     -> 1 dim
        - 'positions' -> 3 dims
        - 'age'       -> 1 dim

    If output_sigmoid=True  -> outputs probability
    If output_sigmoid=False -> outputs embedding (for FiLM)
    """

    def __init__(
        self,
        mlp_features=None,
        hidden_dims=[64, 32],
        output_sigmoid=True
    ):
        super().__init__()

        # Default features if none provided
        if mlp_features is None:
            mlp_features = ['sbref', 'positions', 'age']

        self.mlp_features = mlp_features
        self.output_sigmoid = output_sigmoid
        
        input_dim = 0
        if 'sbref' in self.mlp_features:
            input_dim += 1
        if 'positions' in self.mlp_features:
            input_dim += 3
        if 'age' in self.mlp_features:
            input_dim += 1

        layers = []
        prev_dim = input_dim

        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            prev_dim = h

        if output_sigmoid:
            layers.append(nn.Linear(prev_dim, 1))
            layers.append(nn.Sigmoid())
        else:
            layers.append(nn.Identity())

        self.model = nn.Sequential(*layers)

    def forward(self, x):

        feature_list = []
        idx = 0

        if 'sbref' in self.mlp_features:
            feature_list.append(x[:, idx:idx+1])
            idx += 1

        if 'positions' in self.mlp_features:
            feature_list.append(x[:, idx:idx+3])
            idx += 3

        if 'age' in self.mlp_features:
            feature_list.append(x[:, idx:idx+1])
            idx += 1

        x = torch.cat(feature_list, dim=-1)
        return self.model(x)
