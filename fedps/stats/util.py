def check_channel(channel, send: bool, recv: bool):
    if (send or recv) and channel is None:
        raise ValueError("channel can't be None if send or recv is True")


def check_FL_type(FL_type: str):
    FL_type = FL_type.upper()
    valid_FL_type = {"V", "H"}

    if FL_type in valid_FL_type:
        return FL_type
    else:
        raise ValueError(
            f"Unsupported FL type: {FL_type}, use {valid_FL_type} instead."
        )


def check_role(role: str):
    role = role.lower()
    valid_role = {"client", "server"}

    if role in valid_role:
        return role
    else:
        raise ValueError(f"Unsupported role: {role}, use {valid_role} instead")
