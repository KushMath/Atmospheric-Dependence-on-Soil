function [c_pdf] = bb7Density(u, v, theta, delta)
    % BB7 (Joe-Clayton) copula density
    % theta >= 1, delta > 0
    x = 1 - (1-u).^theta;
    y = 1 - (1-v).^theta;
    S = x.^(-delta) + y.^(-delta) - 1;
    w = S.^(-1./delta);

    c_pdf = (1-u).^(theta-1) .* (1-v).^(theta-1) .* ...
        x.^(-delta-1) .* y.^(-delta-1) .* ...
        S.^(-1./delta - 2) .* (1-w).^(1./theta - 2) .* ...
        (theta.*(1+delta) - (theta.*delta + 1).*w);
end