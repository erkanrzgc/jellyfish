"""Conservative legacy nmap flags. No positional targets, scripts, or file imports."""
import re
import shlex


def validate_ports(value: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError("port listesi eksik")
    for part in value.split(","):
        if not re.fullmatch(r"\d+(?:-\d+)?", part):
            raise ValueError("gecersiz port listesi")
        bounds = [int(v) for v in part.split("-")]
        if any(not 1 <= port <= 65535 for port in bounds) or bounds[0] > bounds[-1]:
            raise ValueError("port aralik disinda")


def nmap_flags(options: str) -> list[str]:
    if not isinstance(options, str):
        raise ValueError("secenekler metin olmali")
    tokens = shlex.split(options)
    flags = {"-sV", "-sT", "-sS", "-sU", "-Pn", "-n", "--open", "--version-light", "-v"}
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token in flags:
            i += 1
            continue
        if token == "-p":
            i += 1
            if i >= len(tokens):
                raise ValueError("-p degeri eksik")
            validate_ports(tokens[i])
        elif token.startswith("-p"):
            validate_ports(token[2:])
        elif token == "--top-ports":
            i += 1
            if i >= len(tokens) or not tokens[i].isdigit() or not 1 <= int(tokens[i]) <= 65535:
                raise ValueError("gecersiz --top-ports")
        else:
            raise ValueError(f"dogrulanmamis nmap secenegi/ek hedef: {token}")
        i += 1
    return tokens
