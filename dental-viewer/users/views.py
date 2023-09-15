import os
import math
import json
import cv2
import numpy as np
from shapely.geometry import Polygon
from django.shortcuts import render, redirect
from django.urls import reverse_lazy
from django.contrib.auth.views import LoginView, PasswordChangeView
from django.contrib.messages.views import SuccessMessageMixin
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.paginator import Paginator
from .forms import LoginForm, ImageForm
from .models import Image


def home(request):
    if request.user.is_authenticated:
        items_per_page = 20
        img_list = Image.objects.filter(user_id=request.user).order_by('-created_at')
        n_pages = math.ceil(len(img_list) / items_per_page)
        pages = list(range(1, n_pages + 1))
        page = request.GET.get('page')
        page = "1" if page not in map(str, pages) else page
        img_list = Paginator(img_list, items_per_page).get_page(page)
        return render(request, 'users/home.html', {"img_list": img_list, "pages": pages, "current_page": int(page)})
    return redirect(to='login')


# Class based view that extends from the built in login view to add a remember me functionality
class CustomLoginView(LoginView):
    form_class = LoginForm

    def form_valid(self, form):
        remember_me = form.cleaned_data.get('remember_me')

        if not remember_me:
            # set session expiry to 0 seconds. So it will automatically close the session after the browser is closed.
            self.request.session.set_expiry(0)

            # Set session as modified to force data updates/cookie to be saved.
            self.request.session.modified = True

        # else browser session will be as long as the session cookie time "SESSION_COOKIE_AGE" defined in settings.py
        return super(CustomLoginView, self).form_valid(form)


class ChangePasswordView(SuccessMessageMixin, PasswordChangeView):
    template_name = 'users/change_password.html'
    success_message = "Successfully Changed Your Password"
    success_url = reverse_lazy('users-home')


def get_bordered_path(path):
    directory, file_name = os.path.split(path)
    base_name, extension = os.path.splitext(file_name)
    new_base_name = base_name + "_bordered"
    new_path = os.path.join(directory, new_base_name + extension)
    return new_path


def get_filled_path(path):
    directory, file_name = os.path.split(path)
    base_name, extension = os.path.splitext(file_name)
    new_base_name = base_name + "_filled"
    new_path = os.path.join(directory, new_base_name + extension)
    return new_path


def segmentation(form_image, form_annotation):
    # Alpha
    alpha = 0.6

    # Colors
    colors = {
        "11": (0.122, 0.467, 0.706),
        "12": (0.682, 0.780, 0.910),
        "13": (1.000, 0.498, 0.055),
        "14": (1.000, 0.733, 0.471),
        "15": (0.173, 0.627, 0.173),
        "16": (0.596, 0.875, 0.541),
        "21": (0.839, 0.153, 0.157),
        "22": (1.000, 0.596, 0.588),
        "23": (0.580, 0.404, 0.741),
        "24": (0.773, 0.690, 0.835),
        "25": (0.549, 0.337, 0.294),
        "26": (0.769, 0.612, 0.580),
        "31": (0.890, 0.467, 0.761),
        "32": (0.969, 0.714, 0.824),
        "33": (0.498, 0.498, 0.498),
        "34": (0.780, 0.780, 0.780),
        "35": (0.737, 0.741, 0.133),
        "36": (0.859, 0.859, 0.553),
        "41": (0.090, 0.745, 0.812),
        "42": (0.620, 0.855, 0.898),
        "43": (0.702, 0.886, 0.804),
        "44": (0.992, 0.804, 0.675),
        "45": (0.957, 0.792, 0.894),
        "46": (0.902, 0.961, 0.788),
    }

    # Image
    original        = cv2.imread("media/" + str(form_image))
    image_filled    = original.copy()
    image_bordered  = original.copy()
    
    # Annotation
    annotation  = json.load(open("media/" + str(form_annotation)))

    # Hulls
    hulls = {}
    for tooth in annotation['tooth']:
        # Label
        label = tooth["teeth_num"]
        # Segmentation
        points = np.array(tooth["segmentation"])
        hull = cv2.convexHull(points)
        hulls[label] = hull
    hulls = dict(sorted(hulls.items(), key=lambda item: item[0]))

    # Cropping
    polys = {}
    for label in hulls.keys():
        poly = Polygon(np.squeeze(hulls[label]))
        for label_ in hulls.keys():
            if label <= label_:
                continue
            poly_ = Polygon(np.squeeze(hulls[label_]))
            poly = poly.difference(poly_)
        polys[label] = poly
    polys = dict(sorted(polys.items(), key=lambda item: item[0], reverse=True))
    
    # Contours
    def add_contours(image, image_type=""):
        if image_type == "bordered":
            color = (
                int(255 * 0.173),
                int(255 * 0.627),
                int(255 * 0.173),
            )
            thickness = 5
        elif image_type == "filled":
            color = (
                int(255 * colors[label][2]),
                int(255 * colors[label][1]),
                int(255 * colors[label][0]),
            )
            thickness = cv2.FILLED
        cv2.drawContours(
            image       = image,
            contours    = [coords_3d],
            contourIdx  = -1,
            color       = color,
            thickness   = thickness,
        )

    # Texts
    def add_texts(image):
        cx = int(np.mean(coords_3d[:, :, 0]) - 50)
        cy = int(np.mean(coords_3d[:, :, 1]) + 30)
        cv2.putText(
            img         = image,
            text        = label,
            org         = (cx, cy),
            fontFace    = cv2.FONT_HERSHEY_SIMPLEX,
            fontScale   = 3,
            color       = (255, 255, 255),
            thickness   = 10,
        )

    # Add contours and texts
    int_coords = lambda x: np.array(x).round().astype(np.int32)
    for label, poly in polys.items():
        coords_3d = np.expand_dims(int_coords(poly.exterior.coords), axis=1)
        # Filled image
        add_contours(image_filled, "filled")
        add_texts(image_filled)
        # Bordered image
        add_contours(image_bordered, "bordered")
        add_texts(image_bordered)

    # Filled
    image_filled = cv2.addWeighted(
        src1    = image_filled, 
        alpha   = alpha, 
        src2    = original, 
        beta    = 1 - alpha, 
        gamma   = 0,
    )

    # Save
    cv2.imwrite("media/" + get_filled_path(str(form_image)), image_filled)
    cv2.imwrite("media/" + get_bordered_path(str(form_image)), image_bordered)


@login_required
def upload(request):
    if request.method == 'POST':
        form = ImageForm(request.POST, request.FILES)
        if form.is_valid():
            post = form.save(commit=False)
            post.user_id = request.user
            form.save()
            segmentation(form.instance.front_image, form.instance.front_ann)
            segmentation(form.instance.left_image, form.instance.left_ann)
            segmentation(form.instance.right_image, form.instance.right_ann)
            segmentation(form.instance.upper_image, form.instance.upper_ann)
            segmentation(form.instance.lower_image, form.instance.lower_ann)
            img_obj = form.instance
            return render(request, 'users/upload.html', {'form': form, 'img_obj': img_obj})
    else:
        form = ImageForm()
        return render(request, 'users/upload.html', {'form': form})
    

@csrf_exempt
@login_required
def ajax_request(request):
    if request.method == 'POST':
        id = request.POST.get('id')
        image = Image.objects.get(id=id)
        os.remove("media/" + get_filled_path(str(image.front_image)))
        os.remove("media/" + get_bordered_path(str(image.front_image)))
        image.front_image.delete()
        os.remove("media/" + get_filled_path(str(image.left_image)))
        os.remove("media/" + get_bordered_path(str(image.left_image)))
        image.left_image.delete()
        os.remove("media/" + get_filled_path(str(image.right_image)))
        os.remove("media/" + get_bordered_path(str(image.right_image)))
        image.right_image.delete()
        os.remove("media/" + get_filled_path(str(image.upper_image)))
        os.remove("media/" + get_bordered_path(str(image.upper_image)))
        image.upper_image.delete()
        os.remove("media/" + get_filled_path(str(image.lower_image)))
        os.remove("media/" + get_bordered_path(str(image.lower_image)))
        image.lower_image.delete()
        image.front_ann.delete()
        image.left_ann.delete()
        image.right_ann.delete()
        image.upper_ann.delete()
        image.lower_ann.delete()
        Image.objects.filter(id=id).delete()
        return JsonResponse({})