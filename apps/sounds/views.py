from django.http import JsonResponse
from django.shortcuts import render

from sounds.utils import lazy_run, speed_run


def sound(request):
    if request.method == "GET":
        return render(request, "compare.html")

    sound_file = request.FILES.get("audio_file", None)
    sound_type = request.POST.get("audio_type")

    data = ""
    if sound_type == "怠速音频":
        data = lazy_run(sound_file)
    elif sound_type == "加速音频":
        data = speed_run(sound_file)
    return JsonResponse(data)
