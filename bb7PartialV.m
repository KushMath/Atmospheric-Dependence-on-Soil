function [h] = bb7PartialV(u, v, theta, delta)
    % BB7 copula: partial derivative of C(u,v) w.r.t. v
    % = P(U <= u | V = v)
    % theta >= 1, delta > 0
    x = 1 - (1-u).^theta;
    y = 1 - (1-v).^theta;
    S = x.^(-delta) + y.^(-delta) - 1;
    w = S.^(-1./delta);

    h = (1-w).^(1./theta - 1) .* S.^(-1./delta - 1) .* ...
        y.^(-delta-1) .* (1-v).^(theta-1);
end