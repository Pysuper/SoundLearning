from django.http import JsonResponse

from sounds.utils import run


def sound(request):
    """
    获取前端音频，返回相似度
    :param request:
    :return:
    """
    result = run(
        wav_file="/home/zheng/Documents/Project/Python/Django/SoundLearning/testfile/Mahlendes.wav",
        model_file="/home/zheng/Documents/Project/Python/Django/SoundLearning/testfile/Mahlendes.h5",
        sr=44100,
        fre_band=(500, 750),
        wav_standard="/home/zheng/Documents/Project/Python/Django/SoundLearning/testfile/M_LSVFT4BR7FN177349驾驶室内 ( 2.50- 4.50 s).wav"
    )
    return JsonResponse(result)
