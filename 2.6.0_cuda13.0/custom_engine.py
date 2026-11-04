from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import List

import paddle

from ppcls.engine.engine import Engine
from ppcls.utils import logger


class CustomEngine(Engine):

    @paddle.no_grad()
    def infer_raw(self, images: List) -> List:
        """
        Runs inferences on the incoming images.

        :param images: the list of images to run inference on (images are in raw bytes)
        :type images: list
        :return: the list of results
        """
        assert self.mode == "infer" and self.eval_mode == "classification"
        results = []
        batch_size = self.config["Infer"]["batch_size"]
        self.model.eval()
        batch_data = []
        for idx, image in enumerate(images):
            try:
                for process in self.preprocess_func:
                    image = process(image)
                batch_data.append(image)
                if len(batch_data) >= batch_size or idx == len(images) - 1:
                    batch_tensor = paddle.to_tensor(batch_data)

                    with self.auto_cast(is_eval=True):
                        out = self.model(batch_tensor)

                    if isinstance(out, list):
                        out = out[0]
                    if isinstance(out, dict) and "Student" in out:
                        out = out["Student"]
                    if isinstance(out, dict) and "logits" in out:
                        out = out["logits"]
                    if isinstance(out, dict) and "output" in out:
                        out = out["output"]

                    result = self.postprocess_func(out, None)
                    results.extend(result)
                    batch_data.clear()
            except Exception as ex:
                logger.error("Exception occurred when processing image #{} with msg: {}".format(idx, ex))
                continue
        return results

