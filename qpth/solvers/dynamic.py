import torch

from qpth.util import get_sizes

def forward(Q, p, G, h, A, b, verbose=0, maxIter=100, dt=0.2):
    """ Solves a given QP problem by modeling it's dynamic representation
        as an RNN with 'maxIter' loops.
    """
    nineq, nz, neq, nBatch = get_sizes(G, A)

    # Base inverses and transposes
    Q_I = torch.inverse(Q)
    G_T = torch.transpose(G, -1, 1)
    A_T = torch.transpose(A, -1, 1)

    # Intermediate matrix expressions
    GQ_I = - G.bmm(Q_I)             # - G Q^{-1}
    AQ_I = - A.bmm(Q_I)             # - A Q^{-1}
    GA = GQ_I.bmm(A_T)              # - G Q^{-1} A^T
    GG = GQ_I.bmm(G_T)              # - G Q^{-1} G^T
    AA = AQ_I.bmm(A_T)              # - A Q^{-1} A^T
    AG = AQ_I.bmm(G_T)              # - A Q^{-1} G^T
    la_d = GQ_I.bmm(p.unsqueeze(2)) - h.unsqueeze(2)       # - G Q^{-1} p - h
    nu_d = AQ_I.bmm(p.unsqueeze(2)) - b.unsqueeze(2)       # - A Q^{-1} p - b

    lams = torch.zeros(nBatch, nineq, 1).type_as(Q).to(Q.device)
    nus = torch.zeros(nBatch, neq, 1).type_as(Q).to(Q.device)
    zeros = torch.zeros(nBatch, nineq, 1).type_as(Q).to(Q.device)

    for _ in range(maxIter):
        dlams = dt * (GG.bmm(lams) + GA.bmm(nus) + la_d)
        dnus = dt * (AG.bmm(lams) + AA.bmm(nus) + nu_d)
        dlams = torch.max(lams + dlams, zeros) - lams
        lams.add_(dlams)
        nus.add_(dnus)

    zhat = - Q_I.bmm(p.unsqueeze(2) + G_T.bmm(lams) + A_T.bmm(nus))
    slacks = h.unsqueeze(2) - G.bmm(zhat)
    
    return zhat.squeeze(2), lams.squeeze(2), nus.squeeze(2), slacks.squeeze(2)

def forward_ineq(Q, p, G, h, verbose=0, maxIter=100, dt=0.2):
    """ Solves a QP problem with only inequality constraints, fastest version
        for the dynamic solver
    """
    nineq, nz, neq, nBatch = get_sizes(G)
    A_T = torch.transpose(G, -1,1)
    Q_I = torch.inverse(Q)

    AQ_I = G.bmm(Q_I)
    D = - AQ_I.bmm(A_T)
    d = - (h.unsqueeze(2) + AQ_I.bmm(p.unsqueeze(2)))

    lams = torch.zeros(nBatch, nineq, 1).type_as(Q).to(Q.device)
    zeros = torch.zeros(nBatch, nineq, 1).type_as(Q).to(Q.device)

    for _ in range(maxIter):
        dlams = dt * (D.bmm(lams) + d)
        dlams = torch.max(lams + dlams, zeros) - lams
        lams.add_(dlams)
        # lams = torch.where(lams > 0, lams, zeros)
        # lams = torch.max(lams + dlams, zeros)

    zhat = - Q_I.bmm(p.unsqueeze(2) + A_T.bmm(lams))
    slacks = h.unsqueeze(2) - G.bmm(zhat)
    
    return zhat.squeeze(2), lams.squeeze(2), slacks.squeeze(2)

def neg(T):
    """ Returns negative elements of a tensor """
    return 0.5 * (T - torch.abs(T))

def forward_eq_new(Q_, p_, G_, h_, A_, b_, verbose=0, maxIter=100, dt=0.2):
    """ Solves equality constraints by dynamically solving both priml and
        dual problems side-by-side
    """
    neq, nz, _, nBatch = get_sizes(A_)

    Q__ = torch.cat((Q_, -Q_), dim=1)
    Q = torch.cat((Q__, -Q__), dim=2)
    p = torch.cat((p_, -p_), dim=1).unsqueeze(2)
    A = torch.cat((A_, -A_), dim=2)
    b = b_.unsqueeze(2)

    Q_T = torch.transpose(Q, -1,1)
    A_T = torch.transpose(A, -1,1)
    p_T = torch.transpose(p, -1,1)
    b_T = torch.transpose(b, -1,1)

    # Expand 'x' to allow for negative values
    x = torch.zeros(nBatch, 2*nz, 1).type_as(Q).to(Q.device)
    y = torch.zeros(nBatch, neq, 1).type_as(Q).to(Q.device)

    for _ in range(maxIter):
        x_T = torch.transpose(x, -1,1)

        g = x_T.bmm(Q).bmm(x) + p_T.bmm(x) - b_T.bmm(y)
        r = neg(Q.bmm(x) + p - A_T.bmm(y))
        dFx = g.bmm(2 * Q.bmm(x) + p)
        dFy = g.bmm(b)
        dx = - dFx - A_T.bmm(A.bmm(x) - b) - neg(x) - Q_T.bmm(r)
        dy = dFy + A.bmm(r)

        x.add_(dx)
        y.add_(dy)

    slacks = torch.zeros(nBatch, neq).type_as(Q).to(Q.device)

    return (x[:,:nz,:] - x[:,nz:,:]).squeeze(2), y.squeeze(2), slacks


def forward_eq_conv(Q, p, G_, h_, A_, b_, verbose=0, maxIter=100, dt=0.2):
    """ DOES NOT WORK, ONLY HERE TO DOCUMENT OUR PROCESS
        Solves the QP problem by transforming equality constraints into two
        inequality constraint, then using the ineq solver.
    """
    nineq, nz, neq, nBatch = get_sizes(G_, A_)

    G = torch.cat((G_, A_, -A_), dim=1)
    h = torch.cat((h_, b_, -b_), dim=1)
    A_T = torch.transpose(G, -1,1)
    Q_I = torch.inverse(Q)

    AQ_I = G.bmm(Q_I)
    D = - AQ_I.bmm(A_T)
    d = - (h.unsqueeze(2) + AQ_I.bmm(p.unsqueeze(2)))

    lams = torch.zeros(nBatch, nineq + neq + neq, 1).type_as(Q).to(Q.device)
    zeros = torch.zeros(nBatch, nineq + neq + neq, 1).type_as(Q).to(Q.device)

    for _ in range(maxIter):
        dlams = dt * (D.bmm(lams) + d)
        dlams = torch.max(lams + dlams, zeros) - lams
        lams.add_(dlams)

    zhat = - Q_I.bmm(p.unsqueeze(2) + A_T.bmm(lams))
    slacks = h.unsqueeze(2) - G.bmm(zhat)
    
    return zhat.squeeze(2), lams.squeeze(2)[:,:nineq], lams.squeeze(2)[:,-neq:], slacks.squeeze(2)
