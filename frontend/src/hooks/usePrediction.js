import { useMutation } from "@tanstack/react-query";
import { postPredict } from "../api/client";

export function usePrediction() {
  const mutation = useMutation({ mutationFn: postPredict });
  return mutation;
}
