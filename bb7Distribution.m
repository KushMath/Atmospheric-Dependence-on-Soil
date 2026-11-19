function [c_cdf] = bb7Distribution(u, v, theta, delta)
    % BB7 (Joe-Clayton) copula CDF
    % theta >= 1, delta > 0
    c_cdf = 1 - (1 - ((1 - (1-u).^theta).^(-delta) + ...
        (1 - (1-v).^theta).^(-delta) - 1).^(-1./delta)).^(1./theta);
end