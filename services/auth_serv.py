from database import User

def get_auth_user_serv(username: str):
    try:
        user = User.select().where(User.username == username)

        return user.dicts().first()
    except Exception as error:
        raise error


def create_user_serv(username: str, password: str):
    try:
        user = User.create(username=username, password=password)
        return user
    except Exception as error:
        raise error