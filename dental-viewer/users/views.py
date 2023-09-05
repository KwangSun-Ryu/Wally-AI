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


def segmentation(form_image, form_annotation):
    alpha = 0.6
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
    original    = cv2.imread("media/" + str(form_image))
    annotation  = json.load(open("media/" + str(form_annotation)))
    image       = original.copy()
    for tooth in annotation['tooth']:
        # Label
        label = tooth["teeth_num"]
        # Segmentation
        points = np.array(tooth["segmentation"])
        contours = cv2.convexHull(points)
        cv2.drawContours(
            image       = image,
            contours    = [contours], 
            contourIdx  = -1, 
            color       = (
                int(255 * colors[label][2]),
                int(255 * colors[label][1]),
                int(255 * colors[label][0]),
            ),
            thickness   = cv2.FILLED,
        )
        # Text
        cx = int(np.mean(contours[:, :, 0]) - 50)
        cy = int(np.mean(contours[:, :, 1]) + 30)
        cv2.putText(
            img = image,
            text = label,
            org= (cx, cy),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=3,
            color=(255, 255, 255),
            thickness=10,
        )
    # Opacity
    image = cv2.addWeighted(
        src1    = image, 
        alpha   = alpha, 
        src2    = original, 
        beta    = 1 - alpha, 
        gamma   = 0,
    )
    # Save
    cv2.imwrite("media/" + str(form_image), image)


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
        image.front_image.delete()
        image.left_image.delete()
        image.right_image.delete()
        image.upper_image.delete()
        image.lower_image.delete()
        image.front_ann.delete()
        image.left_ann.delete()
        image.right_ann.delete()
        image.upper_ann.delete()
        image.lower_ann.delete()
        Image.objects.filter(id=id).delete()
        return JsonResponse({})