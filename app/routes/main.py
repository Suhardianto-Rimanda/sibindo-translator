from flask import Blueprint, render_template

bp = Blueprint("main", __name__)


@bp.route("/")
def index():
    return render_template("index.html", active="penerjemah")


@bp.route("/panduan")
def panduan():
    return render_template("panduan.html", active="panduan")


@bp.route("/tentang")
def tentang():
    return render_template("tentang.html", active="tentang")
