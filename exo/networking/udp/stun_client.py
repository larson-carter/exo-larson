import stun

def get_public_ip_and_port():
    nat_type, external_ip, external_port = stun.get_ip_info()
    return external_ip, external_port

def is_behind_nat():
    nat_type, _, _ = stun.get_ip_info()
    return nat_type != stun.Blocked